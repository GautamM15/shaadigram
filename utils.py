"""
utils.py — shared utility functions used across all pipeline phases.
"""

import json
import logging
import os
import shutil
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS

import config


def setup_logging(phase_name: str) -> logging.Logger:
    """Create and return a logger that writes to logs/{phase_name}_{timestamp}.log.

    Also attaches a StreamHandler so output is echoed to the console.
    The log folder is created if it does not already exist.

    Args:
        phase_name: Short identifier for the phase, e.g. 'phase1_filter'.

    Returns:
        A configured logging.Logger instance.
    """
    os.makedirs(config.LOG_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.LOG_FOLDER, f"{phase_name}_{timestamp}.log")

    logger = logging.getLogger(phase_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called more than once
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging started → %s", log_path)
    return logger


def load_json(path: str) -> dict | list | None:
    """Load and return parsed JSON from *path*.

    Logs an error and returns None if the file is missing or malformed.

    Args:
        path: Absolute or relative path to the JSON file.

    Returns:
        Parsed Python object, or None on failure.
    """
    logger = logging.getLogger(__name__)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("JSON file not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON at %s — %s", path, exc)
        return None


def save_json(path: str, data: dict | list) -> bool:
    """Serialise *data* to JSON and write it to *path*.

    Creates any missing parent directories. Logs an error and returns False
    on failure.

    Args:
        path: Destination file path.
        data: Python dict or list to serialise.

    Returns:
        True on success, False on failure.
    """
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        logger.error("Failed to save JSON to %s — %s", path, exc)
        return False


def get_exif_timestamp(image_path: str) -> datetime | None:
    """Extract the original capture datetime from a photo's EXIF metadata.

    Reads the DateTimeOriginal tag. Falls back to DateTime if
    DateTimeOriginal is absent. Returns None for RAW files or any image
    that carries no readable timestamp.

    Args:
        image_path: Path to the image file.

    Returns:
        datetime object, or None if no timestamp could be extracted.
    """
    logger = logging.getLogger(__name__)
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is None:
                return None

            tag_map = {TAGS.get(k, k): v for k, v in exif_data.items()}

            raw_dt = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
            if not raw_dt:
                return None

            return datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S")
    except Exception as exc:
        logger.warning("Could not read EXIF timestamp from %s — %s", image_path, exc)
        return None


def copy_to_output(src_path: str, dest_folder: str) -> str | None:
    """Copy *src_path* into *dest_folder*, preserving the original filename.

    - Creates *dest_folder* if it does not exist.
    - Never modifies or overwrites the source file.
    - If a file with the same name already exists in *dest_folder*, the copy
      is skipped and the existing path is returned.

    Args:
        src_path: Full path to the source image (on the external drive).
        dest_folder: Destination directory inside the output tree.

    Returns:
        Full path to the copied file, or None on failure.
    """
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(dest_folder, exist_ok=True)
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)

        if os.path.exists(dest_path):
            logger.debug("Already exists, skipping copy: %s", dest_path)
            return dest_path

        shutil.copy2(src_path, dest_path)
        logger.debug("Copied %s → %s", src_path, dest_path)
        return dest_path
    except Exception as exc:
        logger.error("Failed to copy %s to %s — %s", src_path, dest_folder, exc)
        return None
