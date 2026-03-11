"""
Create GFSLT-VLP label files from PHOENIX14T annotations.

GFSLT-VLP uses pickle files containing a dict where each key maps to a sample
with fields: 'name', 'text', 'imgs_path', 'length'.

This script reads the standard PHOENIX14T CSV annotation files and creates
the pickle label files that GFSLT-VLP expects.

Usage:
    python scripts/create_gfslt_labels.py \
        --phoenix_root data/phoenix14t \
        --output_dir baselines/GFSLT-VLP/data/Phonexi-2014T \
        --img_root data/phoenix14t/features/fullFrame-210x260px
"""

import argparse
import csv
import glob
import os
import pickle
from pathlib import Path


def create_labels(
    phoenix_root: str,
    output_dir: str,
    split: str,
    img_root: str = None,
):
    """Create GFSLT-VLP label file for one split.

    Args:
        phoenix_root: Root directory of PHOENIX14T dataset.
        output_dir: Directory to save the label pickle file.
        split: Dataset split ('train', 'dev', 'test').
        img_root: Root for frame images (if different from phoenix_root).
    """
    # Find the annotation CSV
    csv_candidates = [
        os.path.join(phoenix_root, "annotations", f"PHOENIX-2014-T.{split}.corpus.csv"),
        os.path.join(phoenix_root, f"PHOENIX-2014-T.{split}.corpus.csv"),
        os.path.join(phoenix_root, "annotation", f"PHOENIX-2014-T.{split}.corpus.csv"),
    ]

    csv_file = None
    for candidate in csv_candidates:
        if os.path.exists(candidate):
            csv_file = candidate
            break

    if csv_file is None:
        print(f"  ERROR: Could not find annotation CSV for split '{split}'")
        print(f"  Searched: {csv_candidates}")
        return

    # Determine image root
    if img_root is None:
        img_root = os.path.join(phoenix_root, "features", "fullFrame-210x260px")

    print(f"  Reading annotations from: {csv_file}")
    print(f"  Image root: {img_root}")

    # Detect delimiter
    with open(csv_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if "|" in first_line:
        delimiter = "|"
    elif "\t" in first_line:
        delimiter = "\t"
    else:
        delimiter = ","

    data = {}
    skipped = 0

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # Identify column names (handle variations)
        for idx, row in enumerate(reader):
            # Extract fields with flexible column name matching
            name = None
            folder = None
            text = None

            for col_name, value in row.items():
                col_lower = col_name.strip().lower()
                if col_lower in ("id", "name"):
                    name = value.strip()
                elif col_lower == "folder":
                    folder = value.strip()
                elif col_lower in ("translation", "text", "orth"):
                    text = value.strip()

            if name is None:
                name = f"{split}_{idx:05d}"
            if folder is None:
                folder = name
            if text is None:
                print(f"  WARNING: No text found for sample {name}, skipping")
                skipped += 1
                continue

            # Find frame images
            frame_dir_candidates = [
                os.path.join(img_root, split, folder),
                os.path.join(img_root, folder),
            ]

            frame_dir = None
            for candidate in frame_dir_candidates:
                if os.path.exists(candidate):
                    frame_dir = candidate
                    break

            if frame_dir is None:
                print(f"  WARNING: Frame dir not found for {folder}, skipping")
                skipped += 1
                continue

            # List frame files
            frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
            if not frame_files:
                frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

            if not frame_files:
                print(f"  WARNING: No frames found in {frame_dir}, skipping")
                skipped += 1
                continue

            # Create relative paths (GFSLT-VLP prepends img_path config value)
            # Their format: img_path + imgs_path[i] where imgs_path starts with /
            relative_paths = []
            for fp in frame_files:
                rel = os.path.relpath(fp, img_root)
                relative_paths.append("/" + rel.replace("\\", "/"))

            sample_key = f"{split}_{idx:05d}"
            data[sample_key] = {
                "name": name,
                "text": text,
                "imgs_path": relative_paths,
                "length": len(frame_files),
            }

    # Save pickle
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"labels.{split}")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"  Created: {output_path}")
    print(f"  Samples: {len(data)} (skipped: {skipped})")


def main():
    parser = argparse.ArgumentParser(
        description="Create GFSLT-VLP label files from PHOENIX14T annotations"
    )
    parser.add_argument(
        "--phoenix_root",
        required=True,
        help="Root directory of PHOENIX14T dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="baselines/GFSLT-VLP/data/Phonexi-2014T",
        help="Output directory for label pickle files",
    )
    parser.add_argument(
        "--img_root",
        default=None,
        help="Root directory for frame images (default: phoenix_root/features/fullFrame-210x260px)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Creating GFSLT-VLP Label Files")
    print("=" * 60)

    for split in ["train", "dev", "test"]:
        print(f"\nProcessing {split}...")
        create_labels(
            args.phoenix_root,
            args.output_dir,
            split,
            img_root=args.img_root,
        )

    print(f"\nDone! Label files saved to: {args.output_dir}/")
    print("\nNext: Update the GFSLT-VLP config to point to these files.")


if __name__ == "__main__":
    main()
