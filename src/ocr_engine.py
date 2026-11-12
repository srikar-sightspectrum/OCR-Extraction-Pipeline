"""
English printed-text OCR using PaddleOCR 3.x and PP-OCRv6.

This module:
- Initializes the OCR pipeline once.
- Runs OCR on individual images.
- Processes a folder of images.
- Normalizes PaddleOCR output into a stable JSON-friendly structure.
- Validates and saves one JSON file per input image.

Install a compatible PaddlePaddle inference package first, then install the
pinned PaddleOCR package. See the project's installation notes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError as exc:
    raise ImportError(
        "PaddleOCR is not installed. Install a compatible PaddlePaddle "
        "inference package and paddleocr==3.7.0."
    ) from exc


logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}


def initialize_ocr(
    language: str = "en",
    device: str = "cpu",
) -> PaddleOCR:
    """
    Initialize the PaddleOCR 3.x pipeline using PP-OCRv6.

    The returned engine should be reused for all images. Do not call this
    function inside the per-image processing loop.
    """
    if language != "en":
        raise ValueError("This OCR module is configured for English only.")

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string, e.g. 'cpu' or 'gpu:0'.")

    logger.info(
        "Initializing PaddleOCR PP-OCRv6 | language=%s | device=%s",
        language,
        device,
    )

    try:
        engine = PaddleOCR(
            lang=language,
            ocr_version="PP-OCRv6",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception:
        logger.exception("Failed to initialize PaddleOCR.")
        raise

    logger.info("PaddleOCR initialized successfully.")
    return engine


def extract_text(
    image_path: Path,
    ocr_engine: PaddleOCR,
) -> list[Any]:
    """
    Run text detection and recognition on one image.

    Returns the raw list of PaddleOCR 3.x result objects. Each result contains
    fields such as rec_texts, rec_scores, rec_polys and rec_boxes.
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    logger.info("Running OCR on: %s", image_path.name)

    try:
        results = ocr_engine.predict(str(image_path))
        return list(results)
    except Exception:
        logger.exception("OCR inference failed for: %s", image_path)
        raise


def _to_python_list(value: Any) -> list[Any]:
    """Convert NumPy arrays and tuples into JSON-friendly Python lists."""
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return [
            _to_python_list(item) if isinstance(item, (np.ndarray, tuple)) else item
            for item in value
        ]

    return [value]


def format_ocr_results(
    raw_results: Iterable[Any],
    document_name: str,
) -> dict[str, Any]:
    """
    Convert raw PaddleOCR results into the project's JSON structure.

    PaddleOCR 3.x exposes recognized lines through rec_texts and rec_scores.
    rec_polys contains four-point polygons; rec_boxes contains rectangular
    [x_min, y_min, x_max, y_max] boxes. This output preserves polygon geometry.
    """
    pages: list[dict[str, Any]] = []
    total_text_regions = 0

    for result_index, result in enumerate(raw_results, start=1):
        # OCRResult behaves like a dictionary in PaddleOCR 3.x.
        texts = result.get("rec_texts", []) or []
        scores = result.get("rec_scores", []) or []
        polygons = result.get("rec_polys", []) or []

        # Some result configurations may omit rec_polys. Use rec_boxes as a
        # fallback, but preserve their four-value rectangle representation.
        if not polygons:
            polygons = result.get("rec_boxes", []) or []

        texts = _to_python_list(texts)
        scores = _to_python_list(scores)
        polygons = _to_python_list(polygons)

        if not (len(texts) == len(scores) == len(polygons)):
            raise ValueError(
                "PaddleOCR returned mismatched text, score and box counts "
                f"for page result {result_index}."
            )

        text_regions: list[dict[str, Any]] = []

        for region_index, (text, score, polygon) in enumerate(
            zip(texts, scores, polygons),
            start=1,
        ):
            text_regions.append(
                {
                    "id": region_index,
                    "text": str(text),
                    "confidence": float(score),
                    "bounding_box": polygon,
                }
            )

        page_index = result.get("page_index")
        page_number = (
            int(page_index) + 1
            if page_index is not None
            else result_index
        )

        pages.append(
            {
                "page_number": page_number,
                "text_regions": text_regions,
            }
        )
        total_text_regions += len(text_regions)

    return {
        "document_name": Path(document_name).stem,
        "total_text_regions": total_text_regions,
        "pages": pages,
    }


def validate_ocr_results(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate the normalized OCR data structure.

    Empty OCR text is allowed as a result of a page containing no detectable
    text. Confidence must be numeric and between 0 and 1.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return False, ["OCR output must be a dictionary."]

    if not isinstance(data.get("document_name"), str) or not data["document_name"]:
        errors.append("'document_name' must be a non-empty string.")

    pages = data.get("pages")
    if not isinstance(pages, list):
        errors.append("'pages' must be a list.")
        return False, errors

    calculated_total = 0

    for page_position, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            errors.append(f"Page {page_position} must be an object.")
            continue

        regions = page.get("text_regions")
        if not isinstance(regions, list):
            errors.append(f"Page {page_position}: 'text_regions' must be a list.")
            continue

        calculated_total += len(regions)

        for region_position, region in enumerate(regions, start=1):
            prefix = f"Page {page_position}, region {region_position}"

            if not isinstance(region, dict):
                errors.append(f"{prefix} must be an object.")
                continue

            if not isinstance(region.get("text"), str):
                errors.append(f"{prefix}: text must be a string.")

            confidence = region.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                errors.append(f"{prefix}: confidence must be between 0 and 1.")

            box = region.get("bounding_box")
            if not isinstance(box, list) or len(box) not in (4,):
                # The expected normal form is a four-point polygon. A
                # four-value rectangle is also accepted as a fallback.
                if not (
                    isinstance(box, list)
                    and len(box) == 4
                    and all(isinstance(value, (int, float)) for value in box)
                ):
                    errors.append(f"{prefix}: bounding_box has an invalid format.")

    if data.get("total_text_regions") != calculated_total:
        errors.append("'total_text_regions' does not match the page contents.")

    return not errors, errors


def save_ocr_results(
    data: dict[str, Any],
    output_folder: Path,
) -> Path:
    """
    Validate and save OCR data as UTF-8 JSON.

    Raises ValueError if validation fails. The JSON is written to a temporary
    file first, then moved into place to reduce the risk of partial output.
    """
    is_valid, errors = validate_ocr_results(data)
    if not is_valid:
        raise ValueError("Invalid OCR results: " + "; ".join(errors))

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{data['document_name']}.json"
    temporary_path = output_path.with_suffix(".json.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=4)

        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    logger.info("Saved OCR JSON: %s", output_path)
    return output_path


def process_ocr_folder(
    input_folder: Path,
    output_folder: Path,
    ocr_engine: PaddleOCR,
    recursive: bool = False,
) -> dict[str, int]:
    """
    Run OCR on every supported image in a folder and save one JSON per image.

    The OCR engine is passed in so it is initialized once and reused. If
    recursive=True, images in subfolders are discovered too; JSON files are
    still written to the specified output folder using each image's stem.
    """
    input_folder = Path(input_folder).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve()

    if not input_folder.is_dir():
        raise NotADirectoryError(f"OCR input folder does not exist: {input_folder}")

    if input_folder == output_folder:
        raise ValueError("OCR input and JSON output folders must be different.")

    output_folder.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    image_paths = sorted(
        path
        for path in input_folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    summary = {
        "total_images": len(image_paths),
        "successful": 0,
        "failed": 0,
    }

    if not image_paths:
        logger.warning("No supported images found in: %s", input_folder)
        return summary

    for image_path in image_paths:
        try:
            raw_results = extract_text(image_path, ocr_engine)
            structured_results = format_ocr_results(
                raw_results,
                document_name=image_path.name,
            )
            save_ocr_results(structured_results, output_folder)
            summary["successful"] += 1

        except Exception:
            summary["failed"] += 1
            logger.exception("Failed OCR processing for: %s", image_path)

    logger.info("OCR batch summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent

    # CPU is the safe initial default. Change to "gpu:0" only after installing
    # a PaddlePaddle build compatible with the machine's CUDA environment.
    engine = initialize_ocr(
        language="en",
        device="cpu",
    )

    batch_summary = process_ocr_folder(
        input_folder=project_root / "output" / "ocr_images",
        output_folder=project_root / "output" / "json",
        ocr_engine=engine,
        recursive=False,
    )

    logger.info("OCR processing finished: %s", batch_summary)
