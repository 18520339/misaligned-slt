"""
Unit tests for misalignment simulation logic (data/misalign.py).
"""

import pytest

from data.misalign import (
    MisalignmentConfig,
    apply_misalignment,
    compute_max_offset,
    generate_eval_conditions,
    get_condition_name,
    get_condition_type,
)


class TestMisalignmentConfig:
    """Tests for MisalignmentConfig dataclass."""

    def test_default_values(self):
        config = MisalignmentConfig()
        assert config.max_ratio == 0.25
        assert config.max_frames == 50
        assert config.severity_levels == [0.05, 0.10, 0.15, 0.20, 0.25]

    def test_custom_values(self):
        config = MisalignmentConfig(max_ratio=0.3, max_frames=30)
        assert config.max_ratio == 0.3
        assert config.max_frames == 30


class TestComputeMaxOffset:
    """Tests for compute_max_offset()."""

    def test_basic_offset(self):
        # 10% of 100 frames = 10 frames
        assert compute_max_offset(100, 0.10, 0.25, 50) == 10

    def test_capped_by_max_ratio(self):
        # 30% severity, but max_ratio is 0.25 → capped at 25 frames
        assert compute_max_offset(100, 0.30, 0.25, 50) == 25

    def test_capped_by_max_frames(self):
        # 25% of 400 = 100, but max_frames is 50 → capped at 50
        assert compute_max_offset(400, 0.25, 0.25, 50) == 50

    def test_zero_severity(self):
        assert compute_max_offset(100, 0.0, 0.25, 50) == 0

    def test_short_sequence(self):
        # 5% of 10 frames = 0.5, truncated to 0
        assert compute_max_offset(10, 0.05, 0.25, 50) == 0

    def test_rounding(self):
        # 10% of 15 = 1.5, int() truncates to 1
        assert compute_max_offset(15, 0.10, 0.25, 50) == 1


class TestGetConditionName:
    """Tests for get_condition_name()."""

    def test_clean(self):
        assert get_condition_name(0, 0) == "clean"

    def test_head_truncation(self):
        assert get_condition_name(5, 0) == "head_trunc"

    def test_tail_truncation(self):
        assert get_condition_name(0, -5) == "tail_trunc"

    def test_head_contamination(self):
        assert get_condition_name(-5, 0) == "head_contam"

    def test_tail_contamination(self):
        assert get_condition_name(0, 5) == "tail_contam"

    def test_head_trunc_tail_trunc(self):
        assert get_condition_name(5, -5) == "head_trunc+tail_trunc"

    def test_head_trunc_tail_contam(self):
        assert get_condition_name(5, 5) == "head_trunc+tail_contam"

    def test_head_contam_tail_trunc(self):
        assert get_condition_name(-5, -5) == "head_contam+tail_trunc"

    def test_head_contam_tail_contam(self):
        assert get_condition_name(-5, 5) == "head_contam+tail_contam"


class TestGetConditionType:
    """Tests for get_condition_type()."""

    def test_clean(self):
        assert get_condition_type(0, 0) == ("clean", "clean")

    def test_truncated_start(self):
        assert get_condition_type(5, 0) == ("truncated", "clean")

    def test_contaminated_start(self):
        assert get_condition_type(-5, 0) == ("contaminated", "clean")

    def test_truncated_end(self):
        assert get_condition_type(0, -5) == ("clean", "truncated")

    def test_contaminated_end(self):
        assert get_condition_type(0, 5) == ("clean", "contaminated")


class TestGenerateEvalConditions:
    """Tests for generate_eval_conditions()."""

    def test_total_count_46(self):
        """Should generate 46 conditions: 1 clean + 8 sign combos × 5 severities."""
        conditions = generate_eval_conditions(
            T=100, severity_levels=[0.05, 0.10, 0.15, 0.20, 0.25]
        )
        # 1 clean + 8 non-trivial combos × 5 severity levels = 41
        # Actually: 9 sign combos minus (0,0) = 8, × 5 = 40 + 1 clean = 41
        assert len(conditions) == 41

    def test_first_is_clean(self):
        conditions = generate_eval_conditions(T=100)
        assert conditions[0] == (0, 0, "clean", 0.0)

    def test_all_conditions_have_four_elements(self):
        conditions = generate_eval_conditions(T=100)
        for cond in conditions:
            assert len(cond) == 4
            delta_s, delta_e, name, severity = cond
            assert isinstance(delta_s, int)
            assert isinstance(delta_e, int)
            assert isinstance(name, str)
            assert isinstance(severity, float)

    def test_no_duplicate_clean(self):
        """The (0,0) combination should only appear once as the clean baseline."""
        conditions = generate_eval_conditions(T=100)
        clean_count = sum(1 for ds, de, _, _ in conditions if ds == 0 and de == 0)
        assert clean_count == 1

    def test_short_sequence_fewer_conditions(self):
        """Very short sequences may have fewer conditions if offsets round to 0."""
        conditions = generate_eval_conditions(T=5, severity_levels=[0.05])
        # 5% of 5 = 0.25, rounds to 0 → no conditions at this severity
        assert len(conditions) == 1  # Only clean

    def test_offset_magnitudes_correct(self):
        """Check that offset magnitudes match expected values."""
        conditions = generate_eval_conditions(T=100, severity_levels=[0.10])
        for ds, de, name, sev in conditions:
            if name != "clean":
                # At 10% severity with T=100: expected offset = 10
                assert abs(ds) in [0, 10]
                assert abs(de) in [0, 10]

    def test_all_condition_names_present(self):
        """All 8 non-clean condition types should appear."""
        conditions = generate_eval_conditions(T=100, severity_levels=[0.20])
        names = {name for _, _, name, _ in conditions if name != "clean"}
        expected = {
            "head_trunc",
            "tail_trunc",
            "head_contam",
            "tail_contam",
            "head_trunc+tail_trunc",
            "head_trunc+tail_contam",
            "head_contam+tail_trunc",
            "head_contam+tail_contam",
        }
        assert names == expected


class TestApplyMisalignment:
    """Tests for apply_misalignment()."""

    def test_clean_no_change(self):
        """Clean condition should return all current frames unchanged."""
        frame_map, eff_start, eff_end = apply_misalignment(
            num_frames=10, delta_s=0, delta_e=0
        )
        assert len(frame_map) == 10
        assert all(src == "current" for src, _ in frame_map)
        assert [idx for _, idx in frame_map] == list(range(10))
        assert eff_start == 0
        assert eff_end == 10

    def test_head_truncation(self):
        """Head truncation should skip first delta_s frames."""
        frame_map, eff_start, eff_end = apply_misalignment(
            num_frames=10, delta_s=3, delta_e=0
        )
        assert len(frame_map) == 7  # 10 - 3
        assert all(src == "current" for src, _ in frame_map)
        assert frame_map[0] == ("current", 3)
        assert eff_start == 3

    def test_tail_truncation(self):
        """Tail truncation should cut last |delta_e| frames."""
        frame_map, eff_start, eff_end = apply_misalignment(
            num_frames=10, delta_s=0, delta_e=-3
        )
        assert len(frame_map) == 7  # 10 - 3
        assert frame_map[-1] == ("current", 6)
        assert eff_end == 7

    def test_head_contamination_with_prev(self):
        """Head contamination should prepend frames from previous sample."""
        frame_map, eff_start, eff_end = apply_misalignment(
            num_frames=10, delta_s=-3, delta_e=0, prev_num_frames=20
        )
        assert len(frame_map) == 13  # 3 prev + 10 current
        # First 3 frames from prev sample (last 3 of prev)
        assert frame_map[0] == ("prev", 17)
        assert frame_map[1] == ("prev", 18)
        assert frame_map[2] == ("prev", 19)
        # Remaining 10 frames from current
        assert frame_map[3] == ("current", 0)

    def test_tail_contamination_with_next(self):
        """Tail contamination should append frames from next sample."""
        frame_map, eff_start, eff_end = apply_misalignment(
            num_frames=10, delta_s=0, delta_e=3, next_num_frames=20
        )
        assert len(frame_map) == 13  # 10 current + 3 next
        # Last frame of current
        assert frame_map[9] == ("current", 9)
        # First 3 frames from next
        assert frame_map[10] == ("next", 0)
        assert frame_map[11] == ("next", 1)
        assert frame_map[12] == ("next", 2)

    def test_head_contamination_no_prev_uses_black(self):
        """Without previous sample, black frames should be used."""
        frame_map, _, _ = apply_misalignment(
            num_frames=10, delta_s=-3, delta_e=0, prev_num_frames=None
        )
        assert len(frame_map) == 13
        assert frame_map[0] == ("black", 0)
        assert frame_map[1] == ("black", 0)
        assert frame_map[2] == ("black", 0)
        assert frame_map[3] == ("current", 0)

    def test_tail_contamination_no_next_uses_black(self):
        """Without next sample, black frames should be used."""
        frame_map, _, _ = apply_misalignment(
            num_frames=10, delta_s=0, delta_e=3, next_num_frames=None
        )
        assert len(frame_map) == 13
        assert frame_map[10] == ("black", 0)
        assert frame_map[11] == ("black", 0)
        assert frame_map[12] == ("black", 0)

    def test_combined_head_trunc_tail_contam(self):
        """Combined truncation and contamination."""
        frame_map, eff_start, eff_end = apply_misalignment(
            num_frames=20, delta_s=5, delta_e=3, next_num_frames=15
        )
        # Skip first 5, keep remaining 15, append 3 from next = 18
        assert len(frame_map) == 18
        assert frame_map[0] == ("current", 5)
        assert frame_map[-1] == ("next", 2)

    def test_combined_head_contam_tail_trunc(self):
        """Combined contamination and truncation."""
        frame_map, _, _ = apply_misalignment(
            num_frames=20, delta_s=-3, delta_e=-5, prev_num_frames=10
        )
        # Prepend 3 from prev, keep first 15 of current = 18
        assert len(frame_map) == 18
        assert frame_map[0] == ("prev", 7)
        assert frame_map[2] == ("prev", 9)
        assert frame_map[3] == ("current", 0)

    def test_full_truncation(self):
        """Truncating all frames should produce empty output."""
        frame_map, _, _ = apply_misalignment(
            num_frames=10, delta_s=10, delta_e=0
        )
        assert len(frame_map) == 0

    def test_head_contamination_prev_shorter_than_offset(self):
        """Previous sample shorter than contamination offset."""
        frame_map, _, _ = apply_misalignment(
            num_frames=10, delta_s=-5, delta_e=0, prev_num_frames=3
        )
        # Can only take 3 frames from prev (all of them), not 5
        # Actually: start_in_prev = max(0, 3 - 5) = 0, so takes indices [0, 1, 2]
        assert len(frame_map) == 13  # 3 prev + 10 current
        assert frame_map[0] == ("prev", 0)
        assert frame_map[1] == ("prev", 1)
        assert frame_map[2] == ("prev", 2)
