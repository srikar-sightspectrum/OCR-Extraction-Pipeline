"""Run the current image-folder -> PP-StructureV3 -> JSON stage.

Run from the project root with:
    python -m src.main

PDF conversion and image preprocessing are intentionally not invoked in this
phase. The engine reads the already-preprocessed images in output/ocr_images.
"""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.ocr_engine import initialize_ocr, process_ocr_folder

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"
logger = logging.getLogger(__name__)


def load_application_config(config_path: Path = CONFIG_FILE) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")
    for section in ("paths", "ocr"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' is missing or invalid.")

    for key in ("ocr_images", "json_output", "manual_review"):
        value = config["paths"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path string.")

    ocr = config["ocr"]
    if ocr.get("language", "en") != "en":
        raise ValueError("This application is configured for English OCR only.")
    if not isinstance(ocr.get("device", "gpu:0"), str) or not ocr.get("device", "gpu:0").strip():
        raise ValueError("ocr.device must be a non-empty string, e.g. 'gpu:0' or 'cpu'.")
    if not isinstance(ocr.get("recursive", False), bool):
        raise ValueError("ocr.recursive must be a boolean.")
    return config


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def setup_logging(log_folder: Path, level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")
    log_folder.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(
        log_folder / "ocr_application.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=numeric_level, handlers=[file_handler, console_handler], force=True)


def run_pipeline(config: dict[str, Any]) -> dict[str, int]:
    paths = config["paths"]
    input_folder = resolve_project_path(paths["ocr_images"])
    output_folder = resolve_project_path(paths["json_output"])
    review_folder = resolve_project_path(paths["manual_review"])

    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    review_folder.mkdir(parents=True, exist_ok=True)

    ocr_config = config["ocr"]
    engine = initialize_ocr(
        language=ocr_config.get("language", "en"),
        device=ocr_config.get("device", "gpu:0"),
        settings=ocr_config,
    )

    return process_ocr_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        ocr_engine=engine,
        recursive=ocr_config.get("recursive", False),
        manual_review_folder=review_folder,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_application_config()
    setup_logging(PROJECT_ROOT / "logs", config.get("log_level", "INFO"))
    logger.info("Starting image-folder OCR processing.")
    summary = run_pipeline(config)
    logger.info("OCR processing completed: %s", summary)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("OCR application failed.")
        raise
