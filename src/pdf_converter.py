"""Convert PDF documents into page images for the OCR pipeline.

Configuration is read from ``config/config.json`` relative to the project root.
Each page is rendered and saved independently to keep memory usage predictable.
"""

from __future__ import annotations

import json
import logging
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pymupdf


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"
LOG_FOLDER = PROJECT_ROOT / "logs"

SUPPORTED_IMAGE_FORMATS = {"png", "jpg"}
DEFAULT_DPI = 300
DEFAULT_IMAGE_FORMAT = "png"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO") -> None:
    """Configure console and rotating-file logging."""
    level_name = log_level.upper()
    numeric_level = logging.getLevelName(level_name)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {log_level}")

    LOG_FOLDER.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        LOG_FOLDER / "conversion.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric_level,
        handlers=[file_handler, console_handler],
        force=True,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load a JSON configuration file and ensure its root is an object."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Configuration file contains invalid JSON: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a JSON object.")

    return config


def _resolve_config_path(value: Any, setting_name: str) -> Path:
    """Validate a configured path and resolve relative paths from project root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{setting_name}' must be a non-empty path string.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate settings and return a normalized configuration dictionary."""
    for setting in ("input_folder", "output_folder"):
        if setting not in config:
            raise ValueError(f"Missing configuration setting: {setting}")

    input_folder = _resolve_config_path(config["input_folder"], "input_folder")
    output_folder = _resolve_config_path(config["output_folder"], "output_folder")

    if not input_folder.exists() or not input_folder.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")

    if input_folder == output_folder:
        raise ValueError("Input and output folders must be different.")

    dpi = config.get("dpi", DEFAULT_DPI)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("'dpi' must be a positive integer.")

    image_format = config.get("image_format", DEFAULT_IMAGE_FORMAT)
    if not isinstance(image_format, str):
        raise ValueError("'image_format' must be a string.")
    image_format = image_format.lower().strip()
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ValueError(f"Unsupported image format '{image_format}'. Use: {supported}.")

    overwrite_existing = config.get("overwrite_existing", False)
    if not isinstance(overwrite_existing, bool):
        raise ValueError("'overwrite_existing' must be a boolean.")

    recursive = config.get("recursive", False)
    if not isinstance(recursive, bool):
        raise ValueError("'recursive' must be a boolean.")

    log_level = config.get("log_level", "INFO")
    if not isinstance(log_level, str):
        raise ValueError("'log_level' must be a string.")
    if not isinstance(logging.getLevelName(log_level.upper()), int):
        raise ValueError(f"Invalid logging level: {log_level}")

    return {
        **config,
        "input_folder": input_folder,
        "output_folder": output_folder,
        "dpi": dpi,
        "image_format": image_format,
        "overwrite_existing": overwrite_existing,
        "recursive": recursive,
        "log_level": log_level.upper(),
    }


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------

def find_pdf_files(input_folder: Path, recursive: bool = False) -> list[Path]:
    """Return PDF files in a directory, optionally including subdirectories."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_folder.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


# ---------------------------------------------------------------------------
# Single PDF conversion
# ---------------------------------------------------------------------------

def _save_pixmap_safely(
    pixmap: pymupdf.Pixmap,
    image_path: Path,
) -> None:
    """Save an image to a temporary file, then replace the destination."""
    temporary_path: Path | None = None

    try:
        # Create the temporary file in the destination directory so that the
        # final replace operation stays on the same filesystem.
        with tempfile.NamedTemporaryFile(
            prefix=f".{image_path.stem}_",
            suffix=image_path.suffix,
            dir=image_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        pixmap.save(str(temporary_path))
        temporary_path.replace(image_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def convert_single_pdf(
    pdf_path: Path,
    output_folder: Path,
    dpi: int,
    image_format: str,
    overwrite_existing: bool = False,
    relative_folder: Path | None = None,
) -> dict[str, int]:
    """Convert each page of one PDF into an image.

    Returns counts for converted, skipped, and failed pages, plus whether the
    PDF itself could be opened and processed.
    """
    counts = {"converted": 0, "skipped": 0, "failed": 0, "failed_documents": 0}
    target_folder = output_folder / relative_folder if relative_folder else output_folder

    try:
        target_folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create output directory: %s", target_folder)
        counts["failed_documents"] += 1
        return counts

    scale = dpi / 72
    render_matrix = pymupdf.Matrix(scale, scale)

    logger.info("Processing PDF: %s", pdf_path)

    try:
        with pymupdf.open(pdf_path) as pdf_document:
            total_pages = pdf_document.page_count

            if total_pages == 0:
                logger.warning("PDF contains no pages: %s", pdf_path)
                return counts

            for page_index in range(total_pages):
                page_number = page_index + 1
                image_path = target_folder / (
                    f"{pdf_path.stem}_page_{page_number}.{image_format}"
                )

                if image_path.exists() and not overwrite_existing:
                    logger.info("Skipping existing image: %s", image_path)
                    counts["skipped"] += 1
                    continue

                try:
                    page = pdf_document.load_page(page_index)
                    pixmap = page.get_pixmap(matrix=render_matrix, alpha=False)
                    _save_pixmap_safely(pixmap, image_path)

                    counts["converted"] += 1
                    logger.info(
                        "Saved page %d/%d: %s",
                        page_number,
                        total_pages,
                        image_path,
                    )
                except Exception:
                    counts["failed"] += 1
                    logger.exception(
                        "Failed to convert page %d of %s",
                        page_number,
                        pdf_path.name,
                    )

            logger.info(
                "Finished %s | converted=%d skipped=%d failed=%d",
                pdf_path.name,
                counts["converted"],
                counts["skipped"],
                counts["failed"],
            )

    except Exception:
        counts["failed_documents"] += 1
        logger.exception("Could not open or process PDF: %s", pdf_path)

    return counts


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert_pdfs_in_folder_to_images(
    input_folder: Path,
    output_folder: Path,
    dpi: int = DEFAULT_DPI,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    overwrite_existing: bool = False,
    recursive: bool = False,
) -> dict[str, int]:
    """Convert all PDFs in a folder and return a batch processing summary."""
    # Reuse the same validation rules for both config-driven and direct calls.
    normalized = validate_config(
        {
            "input_folder": str(input_folder),
            "output_folder": str(output_folder),
            "dpi": dpi,
            "image_format": image_format,
            "overwrite_existing": overwrite_existing,
            "recursive": recursive,
        }
    )
    input_folder = normalized["input_folder"]
    output_folder = normalized["output_folder"]
    dpi = normalized["dpi"]
    image_format = normalized["image_format"]

    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create output folder: %s", output_folder)
        raise

    pdf_files = find_pdf_files(input_folder, recursive)
    if not pdf_files:
        logger.warning("No PDF files found in: %s", input_folder)
        return {
            "total_pdfs": 0,
            "converted_pages": 0,
            "skipped_pages": 0,
            "failed_pages": 0,
            "failed_documents": 0,
        }

    logger.info("Found %d PDF file(s).", len(pdf_files))

    totals = {
        "converted_pages": 0,
        "skipped_pages": 0,
        "failed_pages": 0,
        "failed_documents": 0,
    }

    for pdf_path in pdf_files:
        relative_folder = (
            pdf_path.parent.relative_to(input_folder) if recursive else None
        )
        result = convert_single_pdf(
            pdf_path=pdf_path,
            output_folder=output_folder,
            dpi=dpi,
            image_format=image_format,
            overwrite_existing=overwrite_existing,
            relative_folder=relative_folder,
        )

        totals["converted_pages"] += result["converted"]
        totals["skipped_pages"] += result["skipped"]
        totals["failed_pages"] += result["failed"]
        totals["failed_documents"] += result["failed_documents"]

    summary = {"total_pdfs": len(pdf_files), **totals}

    logger.info("========== CONVERSION SUMMARY ==========")
    for name, value in summary.items():
        logger.info("%s: %d", name, value)
    logger.info("Batch conversion completed.")

    return summary


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load configuration and run the PDF conversion process."""
    # Establish basic console logging before loading configuration, so startup
    # errors are visible even if the config file is missing or invalid.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        raw_config = load_config(CONFIG_FILE)
        config = validate_config(raw_config)
        setup_logging(config["log_level"])

        logger.info("PDF conversion application started.")
        summary = convert_pdfs_in_folder_to_images(
            input_folder=config["input_folder"],
            output_folder=config["output_folder"],
            dpi=config["dpi"],
            image_format=config["image_format"],
            overwrite_existing=config["overwrite_existing"],
            recursive=config["recursive"],
        )

        if summary["failed_pages"] or summary["failed_documents"]:
            logger.warning("Conversion completed with failures.")
        else:
            logger.info("Application completed successfully.")

    except Exception:
        logger.exception("Application execution failed.")
        raise


if __name__ == "__main__":
    main()