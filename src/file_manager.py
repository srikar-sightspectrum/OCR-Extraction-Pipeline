"""
File and output management utilities for the OCR project.

This module handles filesystem operations only. OCR extraction and result
formatting should remain in ocr_engine.py.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEMP_FILE_PREFIX = ".ocr_tmp_"


def create_output_directories(project_root: str | Path) -> dict[str, Path]:
    """
    Create and return the standard output and log directories.

    The paths are rooted at project_root, so the module does not depend on
    the current working directory. Existing directories are left unchanged.
    """
    root = Path(project_root).expanduser().resolve()

    directories = {
        "ocr_images": root / "output" / "ocr_images",
        "json": root / "output" / "json",
        "manual_review": root / "output" / "manual_review",
        "logs": root / "logs",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def validate_file(
    file_path: str | Path,
    allowed_extensions: Iterable[str] | None = None,
    *,
    require_non_empty: bool = True,
) -> Path:
    """
    Validate that a file exists, is a regular file, has an allowed extension,
    and (by default) is not empty. Return its resolved Path.
    """
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    extensions = (
        {ext.lower() if ext.startswith(".") else f".{ext.lower()}"
         for ext in allowed_extensions}
        if allowed_extensions is not None
        else SUPPORTED_INPUT_EXTENSIONS
    )

    if path.suffix.lower() not in extensions:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Allowed extensions: {', '.join(sorted(extensions))}"
        )

    if require_non_empty and path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")

    # Opening the file verifies that it is accessible for reading.
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        raise OSError(f"File is not readable: {path}") from exc

    return path.resolve()


def get_output_path(
    input_path: str | Path,
    output_folder: str | Path,
    *,
    suffix: str = "",
    extension: str = ".json",
    relative_to: str | Path | None = None,
) -> Path:
    """
    Build an output path using the input file's name.

    If relative_to is supplied and input_path is inside that directory,
    its subfolder structure is preserved under output_folder.
    """
    source = Path(input_path)
    output_root = Path(output_folder).expanduser().resolve()

    if not extension.startswith("."):
        extension = f".{extension}"

    relative_parent = Path()
    if relative_to is not None:
        base = Path(relative_to).expanduser().resolve()
        try:
            relative_parent = source.resolve().parent.relative_to(base)
        except ValueError:
            logger.warning(
                "Input path %s is outside %s; saving directly in output folder.",
                source,
                base,
            )

    destination_folder = output_root / relative_parent
    destination_folder.mkdir(parents=True, exist_ok=True)

    return destination_folder / f"{source.stem}{suffix}{extension.lower()}"


def _atomic_write_text(output_path: Path, content: str) -> None:
    """Write text to a temporary file, then atomically replace the destination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=TEMP_FILE_PREFIX,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, output_path)
    except OSError:
        logger.exception("Could not write output file: %s", output_path)
        raise
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                logger.warning(
                    "Could not remove temporary file: %s",
                    temporary_path,
                    exc_info=True,
                )


def save_json(
    data: dict[str, Any] | list[Any],
    output_path: str | Path,
    *,
    indent: int = 4,
    overwrite: bool = True,
) -> Path:
    """
    Validate and save Python data as UTF-8 JSON.

    Writes through a temporary file to reduce the risk of leaving a
    partially written JSON file if the process is interrupted.
    """
    destination = Path(output_path).expanduser().resolve()

    if destination.suffix.lower() != ".json":
        raise ValueError(f"Output path must use the .json extension: {destination}")

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination}")

    try:
        serialized_data = json.dumps(
            data,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Data cannot be serialized as valid JSON.") from exc

    _atomic_write_text(destination, serialized_data + "\n")
    logger.info("Saved JSON output: %s", destination)
    return destination


def save_processing_summary(
    summary: dict[str, Any],
    output_folder: str | Path,
    *,
    filename: str = "processing_summary.json",
    overwrite: bool = True,
) -> Path:
    """
    Save run-level counts and processing information as a JSON summary.

    The caller is responsible for defining the summary fields and counts.
    """
    if Path(filename).name != filename or not filename.lower().endswith(".json"):
        raise ValueError("filename must be a simple .json filename.")

    summary_path = Path(output_folder).expanduser().resolve() / filename
    return save_json(summary, summary_path, overwrite=overwrite)


def cleanup_temporary_files(
    directories: Iterable[str | Path],
) -> int:
    """
    Remove only temporary files created by this module.

    Files are identified by the dedicated '.ocr_tmp_' prefix. Other .tmp
    files and all normal user files are left untouched.
    """
    removed_count = 0

    for directory_value in directories:
        directory = Path(directory_value).expanduser()

        if not directory.exists():
            continue
        if not directory.is_dir():
            logger.warning("Skipping non-directory during cleanup: %s", directory)
            continue

        for temporary_file in directory.glob(f"{TEMP_FILE_PREFIX}*.tmp"):
            try:
                if temporary_file.is_file():
                    temporary_file.unlink()
                    removed_count += 1
            except OSError:
                logger.warning(
                    "Could not remove temporary file: %s",
                    temporary_file,
                    exc_info=True,
                )

    logger.info("Removed %d OCR temporary file(s).", removed_count)
    return removed_count
