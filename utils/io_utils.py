"""
utils/io_utils.py
=================
File I/O helpers: directory setup, image loading, image saving.
"""

import os
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.debug("Ensured directory: %s", d)


def collect_images(folder: str, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[str]:
    """
    Recursively collect all image file paths under *folder*,
    sorted alphabetically, skipping macOS metadata folders.
    """
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        if "__MACOSX" in root:
            continue
        for f in sorted(files):
            if f.startswith("."):
                continue
            if f.lower().endswith(extensions):
                paths.append(os.path.join(root, f))
    return paths


def load_gray(path: str) -> Optional[np.ndarray]:
    """Load image as grayscale. Returns None and logs a warning on failure."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("Could not load image: %s", path)
    return img


def load_bgr(path: str) -> Optional[np.ndarray]:
    """Load image as BGR colour. Returns None and logs a warning on failure."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Could not load image: %s", path)
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    """Save image to *path*. Returns True on success."""
    ok = cv2.imwrite(path, img)
    if ok:
        logger.debug("Saved: %s", path)
    else:
        logger.error("Failed to save: %s", path)
    return ok
