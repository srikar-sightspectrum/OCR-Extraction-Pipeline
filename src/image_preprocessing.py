"""
Basic adaptive image preprocessing for the OCR pipeline.

Workflow:
1. Read an image and convert it to grayscale.
2. Measure blur, brightness, and contrast on a smaller assessment copy.
3. Leave good-quality images unchanged.
4. Apply CLAHE only for low contrast.
5. Apply a mild brightness adjustment only for very dark or bright images.
6. Route images with a very low blur score to manual review.

Thresholds are starting values. Calibrate them against representative documents
from your own OCR workload before relying on them in production.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}


@dataclass(frozen=True)
class QualityThresholds:
    """Initial image-quality thresholds; validate these on real documents."""

    low_brightness: float = 80.0
    high_brightness: float = 220.0
    low_contrast: float = 40.0
    severe_blur: float = 50.0


def calculate_blur(gray_image: np.ndarray) -> float:
    """Return variance of the Laplacian as a basic sharpness indicator."""
    if gray_image.ndim != 2:
        raise ValueError("calculate_blur expects a grayscale image.")

    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def calculate_brightness(gray_image: np.ndarray) -> float:
    """Return mean grayscale pixel intensity (0=black, 255=white)."""
    if gray_image.ndim != 2:
        raise ValueError("calculate_brightness expects a grayscale image.")

    return float(np.mean(gray_image))


def calculate_contrast(gray_image: np.ndarray) -> float:
    """Return standard deviation of grayscale pixel intensity."""
    if gray_image.ndim != 2:
        raise ValueError("calculate_contrast expects a grayscale image.")

    return float(np.std(gray_image))


def assess_image_quality(
    image: np.ndarray,
    assessment_width: int = 750,
) -> dict[str, float]:
    """
    Measure image quality on a resized copy.

    The original image is not modified. Resizing reduces the cost of computing
    the metrics while preserving the aspect ratio.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if assessment_width <= 0:
        raise ValueError("assessment_width must be a positive integer.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected a grayscale or 3-channel BGR image.")

    height, width = gray.shape
    if width > assessment_width:
        resized_height = max(1, round(height * assessment_width / width))
        assessment_image = cv2.resize(
            gray,
            (assessment_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        assessment_image = gray

    return {
        "blur_score": calculate_blur(assessment_image),
        "brightness": calculate_brightness(assessment_image),
        "contrast": calculate_contrast(assessment_image),
    }


def preprocess_image(
    image: np.ndarray,
    thresholds: QualityThresholds | None = None,
) -> tuple[np.ndarray | None, str, dict[str, float]]:
    """
    Assess and conditionally preprocess one image.

    Returns:
        (output_image, status, metrics)

        status is one of:
        - DIRECT_TO_OCR
        - PROCESSED_TO_OCR
        - MANUAL_REVIEW

        output_image is None only for MANUAL_REVIEW.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    thresholds = thresholds or QualityThresholds()

    if image.ndim == 2:
        gray_image = image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected a grayscale or 3-channel BGR image.")

    metrics = assess_image_quality(gray_image)

    logger.info(
        "Image quality | blur=%.2f brightness=%.2f contrast=%.2f",
        metrics["blur_score"],
        metrics["brightness"],
        metrics["contrast"],
    )

    # A very low blur score is a warning, not proof that an image is unusable.
    # This conservative rule routes it for human inspection rather than
    # attempting an unreliable automatic deblurring operation.
    if metrics["blur_score"] < thresholds.severe_blur:
        logger.warning(
            "Very low blur score (%.2f); routing image to manual review.",
            metrics["blur_score"],
        )
        return None, "MANUAL_REVIEW", metrics

    processed_image = gray_image.copy()
    corrections_applied = False

    # Apply local contrast enhancement only when measured contrast is low.
    if metrics["contrast"] < thresholds.low_contrast:
        logger.info("Applying CLAHE because contrast is below threshold.")
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        processed_image = clahe.apply(processed_image)
        corrections_applied = True

    # Apply a mild brightness adjustment only when the original assessment
    # indicates an unusually dark or bright page.
    if metrics["brightness"] < thresholds.low_brightness:
        logger.info("Applying mild brightness correction (dark image).")
        processed_image = cv2.convertScaleAbs(
            processed_image,
            alpha=1.0,
            beta=20,
        )
        corrections_applied = True

    elif metrics["brightness"] > thresholds.high_brightness:
        logger.info("Applying mild brightness correction (bright image).")
        processed_image = cv2.convertScaleAbs(
            processed_image,
            alpha=1.0,
            beta=-20,
        )
        corrections_applied = True

    status = "PROCESSED_TO_OCR" if corrections_applied else "DIRECT_TO_OCR"
    return processed_image, status, metrics


def _save_image(image: np.ndarray, destination: Path) -> None:
    """Save an image through a temporary file in the destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}_",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        if not cv2.imwrite(str(temporary_path), image):
            raise OSError(f"OpenCV could not write image: {destination}")

        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def process_image_folder(
    input_folder: Path,
    output_folder: Path,
    manual_review_folder: Path,
    thresholds: QualityThresholds | None = None,
    recursive: bool = False,
) -> dict[str, int]:
    """
    Process supported images in a folder.

    OCR-ready images are saved as PNG. Images routed to manual review are
    copied in their original format. With recursive=True, subfolder structure
    is preserved in both output locations.
    """
    input_folder = Path(input_folder).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve()
    manual_review_folder = Path(manual_review_folder).expanduser().resolve()

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")

    if input_folder in (output_folder, manual_review_folder):
        raise ValueError("Input folder must differ from output folders.")

    if output_folder == manual_review_folder:
        raise ValueError("OCR output and manual-review folders must differ.")

    output_folder.mkdir(parents=True, exist_ok=True)
    manual_review_folder.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    image_paths = sorted(
        path
        for path in input_folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    summary = {
        "total_images": len(image_paths),
        "direct_to_ocr": 0,
        "processed_to_ocr": 0,
        "manual_review": 0,
        "errors": 0,
    }

    if not image_paths:
        logger.warning("No supported images found in %s", input_folder)
        return summary

    thresholds = thresholds or QualityThresholds()
    logger.info("Found %d image(s) to process.", len(image_paths))

    for image_path in image_paths:
        relative_path = image_path.relative_to(input_folder)

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            final_image, status, metrics = preprocess_image(image, thresholds)

            if status == "MANUAL_REVIEW":
                destination = manual_review_folder / relative_path
                _save_image(image, destination)
                summary["manual_review"] += 1
                logger.info("Sent to manual review: %s", destination)
                continue

            # Use PNG for a lossless OCR-ready output.
            destination = (output_folder / relative_path).with_suffix(".png")
            _save_image(final_image, destination)

            if status == "DIRECT_TO_OCR":
                summary["direct_to_ocr"] += 1
            else:
                summary["processed_to_ocr"] += 1

            logger.info(
                "Saved %s | status=%s | metrics=%s",
                destination,
                status,
                metrics,
            )

        except (OSError, ValueError, cv2.error):
            summary["errors"] += 1
            logger.exception("Failed to process image: %s", image_path)
        except Exception:
            summary["errors"] += 1
            logger.exception("Unexpected error processing image: %s", image_path)

    logger.info("Image preprocessing summary: %s", summary)
    return summary


if __name__ == "__main__":
    # Adjust these paths to match your project structure.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent

    result = process_image_folder(
        input_folder=project_root / "images_from_pdf",
        output_folder=project_root / "ocr_ready_images",
        manual_review_folder=project_root / "manual_review",
    )

    print(result)
