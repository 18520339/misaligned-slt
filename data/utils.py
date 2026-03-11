"""
Data utilities for PHOENIX14T dataset.

Handles frame I/O, annotation parsing, and German text preprocessing.
"""

import csv
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_frame_paths(frame_dir: str) -> List[str]:
    """Load and sort all frame file paths from a directory.

    PHOENIX14T stores frames as individual image files with names like
    images0001.png, images0002.png, etc.

    Args:
        frame_dir: Path to the directory containing frame images.

    Returns:
        Sorted list of absolute paths to frame files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    # Support common image formats
    extensions = {".png", ".jpg", ".jpeg"}
    frame_paths = sorted(
        [str(f) for f in frame_dir.iterdir() if f.suffix.lower() in extensions]
    )
    return frame_paths


def load_frame_image(
    path: str,
    transform: Optional[Callable] = None,
) -> Any:
    """Load a single frame image.

    Args:
        path: Path to the image file.
        transform: Optional transform to apply (e.g., torchvision transform).

    Returns:
        Image as PIL Image or transformed tensor.
    """
    img = Image.open(path).convert("RGB")
    if transform is not None:
        img = transform(img)
    return img


def create_black_frame(height: int = 260, width: int = 210) -> Image.Image:
    """Create a black (zero-filled) frame for padding.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        Black PIL Image.
    """
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def get_phoenix_annotations(
    split: str,
    data_root: str,
    annotation_dir: str = "annotations",
    annotation_file_pattern: str = "PHOENIX-2014-T.{split}.corpus.csv",
) -> List[Dict[str, Any]]:
    """Parse PHOENIX14T annotation file for a given split.

    The CSV has columns: id, folder, signer, comment, start, end, gloss, text
    (tab-separated in the official format).

    Args:
        split: Dataset split ("train", "dev", "test").
        data_root: Root directory of the PHOENIX14T dataset.
        annotation_dir: Subdirectory containing annotation files.
        annotation_file_pattern: Filename pattern with {split} placeholder.

    Returns:
        List of annotation dicts with keys: id, folder, signer, gloss, text.
    """
    ann_file = os.path.join(
        data_root, annotation_dir, annotation_file_pattern.format(split=split)
    )

    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    annotations = []
    with open(ann_file, "r", encoding="utf-8") as f:
        # PHOENIX14T uses pipe '|' as delimiter in some versions, tab in others
        # Try to detect the delimiter
        first_line = f.readline()
        f.seek(0)

        if "|" in first_line:
            delimiter = "|"
        elif "\t" in first_line:
            delimiter = "\t"
        else:
            delimiter = ","

        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # Normalize column names (handle variations in the CSV headers)
            ann = {}
            for key, value in row.items():
                key_lower = key.strip().lower()
                if key_lower in ("id", "name"):
                    ann["id"] = value.strip()
                elif key_lower == "folder":
                    ann["folder"] = value.strip()
                elif key_lower == "signer":
                    ann["signer"] = value.strip()
                elif key_lower == "gloss":
                    ann["gloss"] = value.strip()
                elif key_lower in ("text", "translation", "orth"):
                    ann["text"] = value.strip()

            # Ensure required fields exist
            if "id" not in ann or "text" not in ann:
                continue

            # Use folder as id if folder is present but id is not useful
            if "folder" not in ann:
                ann["folder"] = ann["id"]

            annotations.append(ann)

    return annotations


def get_frame_dir(
    data_root: str,
    split: str,
    sample_folder: str,
    frame_dir_pattern: str = "features/fullFrame-210x260px/{split}/{sample_id}",
) -> str:
    """Construct the frame directory path for a sample.

    Args:
        data_root: Root directory of the PHOENIX14T dataset.
        split: Dataset split.
        sample_folder: Sample folder name (from annotations).
        frame_dir_pattern: Pattern with {split} and {sample_id} placeholders.

    Returns:
        Absolute path to the frame directory.
    """
    frame_dir = frame_dir_pattern.format(split=split, sample_id=sample_folder)
    return os.path.join(data_root, frame_dir)


def preprocess_text(text: str, lowercase: bool = True) -> str:
    """Preprocess German text following PHOENIX14T evaluation conventions.

    Args:
        text: Raw German text.
        lowercase: Whether to lowercase the text.

    Returns:
        Preprocessed text string.
    """
    if lowercase:
        text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def count_frames_in_dir(frame_dir: str) -> int:
    """Count the number of frame files in a directory without loading them.

    Args:
        frame_dir: Path to the frame directory.

    Returns:
        Number of image files in the directory, or 0 if directory doesn't exist.
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        return 0

    extensions = {".png", ".jpg", ".jpeg"}
    return sum(1 for f in frame_dir.iterdir() if f.suffix.lower() in extensions)
