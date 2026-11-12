"""
Main entry point for the OCR application.

Run from the project root with:
    python -m src.main

Pipeline:
    PDF files -> page images -> image preprocessing -> PaddleOCR -> JSON
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.pdf_converter import convert_pdfs_in_folder_to_images
from src.image_preprocessing import (
    QualityThresholds,
    process_image_folder,
)
from src.ocr_engine import initialize_ocr, process_ocr_folder
from src.file_manager import (
    create_output_directories,
    save_processing_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"

logger = logging.getLogger(__name__)


def load_application_config(config_path: Path = CONFIG_FILE) -> dict[str, Any]:
    """Load and validate the application configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")

    required_sections = ("paths", "pdf", "preprocessing", "ocr")
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' is missing or invalid.")

    paths = config["paths"]
    required_paths = (
        "input_folder",
        "pdf_images",
        "ocr_images",
        "json_output",
        "manual_review",
    )
    for key in required_paths:
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path string.")

    pdf_config = config["pdf"]
    dpi = pdf_config.get("dpi", 300)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("pdf.dpi must be a positive integer.")

    image_format = pdf_config.get("image_format", "png")
    if not isinstance(image_format, str) or image_format.lower() not in {"png", "jpg"}:
        raise ValueError("pdf.image_format must be 'png' or 'jpg'.")

    for key in ("overwrite_existing", "recursive"):
        if not isinstance(pdf_config.get(key, False), bool):
            raise ValueError(f"pdf.{key} must be a boolean.")

    ocr_config = config["ocr"]
    if ocr_config.get("language", "en") != "en":
        raise ValueError("This application is configured for English OCR only.")

    device = ocr_config.get("device", "cpu")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("ocr.device must be a non-empty string.")

    return config


def setup_application_logging(
    log_folder: Path,
    log_level: str = "INFO",
) -> None:
    """Configure console and rotating-file logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {log_level}")

    log_folder.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_folder / "application.log",
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


def resolve_project_path(value: str) -> Path:
    """Resolve configured paths relative to the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def initialize_application(config: dict[str, Any]) -> tuple[dict[str, Path], Any]:
    """Create application directories and initialize the OCR engine."""
    paths_config = config["paths"]

    directories = {
        "input": resolve_project_path(paths_config["input_folder"]),
        "pdf_images": resolve_project_path(paths_config["pdf_images"]),
        "ocr_images": resolve_project_path(paths_config["ocr_images"]),
        "json": resolve_project_path(paths_config["json_output"]),
        "manual_review": resolve_project_path(paths_config["manual_review"]),
        "logs": PROJECT_ROOT / "logs",
    }

    directories["input"].mkdir(parents=True, exist_ok=True)

    # Use the shared file-manager function for the standard output directories.
    create_output_directories(PROJECT_ROOT)

    for key, directory in directories.items():
        if key != "input":
            directory.mkdir(parents=True, exist_ok=True)

    ocr_engine = initialize_ocr(
        language=config["ocr"].get("language", "en"),
        device=config["ocr"].get("device", "cpu"),
    )

    return directories, ocr_engine


def process_pdf_stage(
    config: dict[str, Any],
    directories: dict[str, Path],
) -> dict[str, int]:
    """Convert all PDFs in the input folder into page images."""
    pdf_config = config["pdf"]

    logger.info("Starting PDF conversion stage.")

    summary = convert_pdfs_in_folder_to_images(
        input_folder=directories["input"],
        output_folder=directories["pdf_images"],
        dpi=pdf_config.get("dpi", 300),
        image_format=pdf_config.get("image_format", "png").lower(),
        overwrite_existing=pdf_config.get("overwrite_existing", False),
        recursive=pdf_config.get("recursive", False),
    )

    logger.info("PDF conversion stage completed: %s", summary)
    return summary


def process_preprocessing_stage(
    config: dict[str, Any],
    directories: dict[str, Path],
) -> dict[str, int]:
    """Assess converted page images and apply only basic corrections."""
    preprocessing_config = config["preprocessing"]

    thresholds = QualityThresholds(
        low_brightness=preprocessing_config.get("low_brightness", 80.0),
        high_brightness=preprocessing_config.get("high_brightness", 220.0),
        low_contrast=preprocessing_config.get("low_contrast", 40.0),
        severe_blur=preprocessing_config.get("severe_blur", 50.0),
    )

    logger.info("Starting image preprocessing stage.")

    summary = process_image_folder(
        input_folder=directories["pdf_images"],
        output_folder=directories["ocr_images"],
        manual_review_folder=directories["manual_review"],
        thresholds=thresholds,
        recursive=preprocessing_config.get("recursive", False),
    )

    logger.info("Image preprocessing stage completed: %s", summary)
    return summary


def process_ocr_stage(
    config: dict[str, Any],
    directories: dict[str, Path],
    ocr_engine: Any,
) -> dict[str, int]:
    """Run English text detection and recognition on OCR-ready images."""
    logger.info("Starting OCR stage.")

    summary = process_ocr_folder(
        input_folder=directories["ocr_images"],
        output_folder=directories["json"],
        ocr_engine=ocr_engine,
        recursive=config["ocr"].get("recursive", False),
    )

    logger.info("OCR stage completed: %s", summary)
    return summary


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run PDF conversion, preprocessing and OCR in sequence."""
    directories, ocr_engine = initialize_application(config)

    # Stop the pipeline if an earlier stage fails at application level.
    pdf_summary = process_pdf_stage(config, directories)
    preprocessing_summary = process_preprocessing_stage(config, directories)
    ocr_summary = process_ocr_stage(config, directories, ocr_engine)

    final_summary = {
        "pdf_conversion": pdf_summary,
        "image_preprocessing": preprocessing_summary,
        "ocr": ocr_summary,
    }

    save_processing_summary(
        summary=final_summary,
        output_folder=directories["json"],
        filename="processing_summary.json",
    )

    logger.info("Complete OCR pipeline finished.")
    logger.info("Final summary: %s", final_summary)

    return final_summary


def main() -> None:
    """Application entry point."""
    # Basic logging ensures configuration errors are visible before the
    # rotating file logger has been initialized.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    try:
        config = load_application_config()
        setup_application_logging(
            log_folder=PROJECT_ROOT / "logs",
            log_level=config.get("log_level", "INFO"),
        )

        logger.info("OCR application started.")
        run_pipeline(config)

    except Exception:
        logger.exception("OCR application failed.")
        raise


if __name__ == "__main__":
    main()
