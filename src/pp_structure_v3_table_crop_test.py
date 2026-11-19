"""PP-StructureV3 isolated invoice-table test.

This version crops the line-item table from the preprocessed invoice image
before passing it to PP-StructureV3. Crop coordinates are relative to the
source image, so they can be adjusted without changing the processing code.

Model identifiers are retained from the supplied test script. Their
availability/compatibility must be confirmed in the local PaddleX environment.
"""

import logging
from pathlib import Path

from PIL import Image
from paddleocr import PPStructureV3


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_IMAGE = Path(
    r"C:\OCR_PROJECT\output\ocr_images\27th oct_page_1.png"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "pp_structure_v3_table_crop_test"
CROP_DIR = OUTPUT_DIR / "input_crop"
JSON_DIR = OUTPUT_DIR / "json"
VISUAL_DIR = OUTPUT_DIR / "visualizations"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
HTML_DIR = OUTPUT_DIR / "html"
EXCEL_DIR = OUTPUT_DIR / "excel"

DEVICE = "cpu"

# Relative crop coordinates: (left, top, right, bottom).
# Initial values are estimated for the supplied 27-Oct invoice image.
# They include the table column headings and item rows, and exclude totals.
CROP_RELATIVE = (0.035, 0.270, 0.995, 0.565)

TEXT_DETECTION_MODEL = "PP-OCRv6_small_det"
TEXT_RECOGNITION_MODEL = "PP-OCRv6_small_rec"

TABLE_CLASSIFICATION_MODEL = "PP-LCNet_x1_0_table_cls"
WIRED_TABLE_STRUCTURE_MODEL = "SLANeXt_wired"
WIRELESS_TABLE_STRUCTURE_MODEL = "SLANeXt_wireless"
WIRED_TABLE_CELLS_MODEL = "RT-DETR-L_wired_table_cell_det"
WIRELESS_TABLE_CELLS_MODEL = "RT-DETR-L_wireless_table_cell_det"


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# CROP TABLE REGION
# --------------------------------------------------

def create_table_crop(image_path: Path) -> Path:
    """Create and save a crop using normalized image coordinates."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    CROP_DIR.mkdir(parents=True, exist_ok=True)
    crop_path = CROP_DIR / f"{image_path.stem}_table_crop.png"

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size

        left_r, top_r, right_r, bottom_r = CROP_RELATIVE

        if not (
            0 <= left_r < right_r <= 1
            and 0 <= top_r < bottom_r <= 1
        ):
            raise ValueError(
                "CROP_RELATIVE must contain valid normalized "
                "(left, top, right, bottom) coordinates."
            )

        left = round(left_r * width)
        top = round(top_r * height)
        right = round(right_r * width)
        bottom = round(bottom_r * height)

        crop = image.crop((left, top, right, bottom))
        crop.save(crop_path)

    logger.info("Source image dimensions: %s x %s", width, height)
    logger.info(
        "Crop pixel coordinates: left=%s, top=%s, right=%s, bottom=%s",
        left, top, right, bottom,
    )
    logger.info("Crop dimensions: %s x %s", right - left, bottom - top)
    logger.info("Saved table crop: %s", crop_path)

    return crop_path


# --------------------------------------------------
# INITIALIZE ONE PP-STRUCTURE-V3 PIPELINE
# --------------------------------------------------

def initialize_pipeline():
    logger.info("Initializing PP-StructureV3 on %s...", DEVICE)
    logger.info("OCR detection model: %s", TEXT_DETECTION_MODEL)
    logger.info("OCR recognition model: %s", TEXT_RECOGNITION_MODEL)
    logger.info("Wired table structure model: %s", WIRED_TABLE_STRUCTURE_MODEL)
    logger.info("Wireless table structure model: %s", WIRELESS_TABLE_STRUCTURE_MODEL)
    logger.info("Wired cell detection model: %s", WIRED_TABLE_CELLS_MODEL)
    logger.info("Wireless cell detection model: %s", WIRELESS_TABLE_CELLS_MODEL)

    pipeline = PPStructureV3(
        device=DEVICE,
        text_detection_model_name=TEXT_DETECTION_MODEL,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL,
        text_recognition_batch_size=1,

        use_table_recognition=True,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_MODEL,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_MODEL,
        wired_table_cells_detection_model_name=WIRED_TABLE_CELLS_MODEL,
        wireless_table_cells_detection_model_name=WIRELESS_TABLE_CELLS_MODEL,

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_region_detection=False,
    )

    logger.info("PP-StructureV3 initialized successfully.")
    return pipeline


# --------------------------------------------------
# PROCESS CROPPED TABLE
# --------------------------------------------------

def process_image(pipeline, image_path: Path):
    logger.info("Running PP-StructureV3 on cropped table: %s", image_path)
    results = pipeline.predict(input=str(image_path))

    if not results:
        raise RuntimeError("PP-StructureV3 returned no results.")

    for index, result in enumerate(results, start=1):
        logger.info("Saving result %s", index)

        result.save_to_json(save_path=str(JSON_DIR))
        result.save_to_img(save_path=str(VISUAL_DIR))
        result.save_to_markdown(save_path=str(MARKDOWN_DIR))

        if hasattr(result, "save_to_html"):
            try:
                result.save_to_html(save_path=str(HTML_DIR))
            except Exception as exc:
                logger.warning("HTML export skipped: %s", exc)

        if hasattr(result, "save_to_xlsx"):
            try:
                result.save_to_xlsx(save_path=str(EXCEL_DIR))
            except Exception as exc:
                logger.warning("Excel export skipped: %s", exc)

    logger.info("Processing completed. Result count: %d", len(results))
    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    for folder in (
        CROP_DIR, JSON_DIR, VISUAL_DIR, MARKDOWN_DIR, HTML_DIR, EXCEL_DIR
    ):
        folder.mkdir(parents=True, exist_ok=True)

    # Step 1: isolate the line-item table.
    crop_path = create_table_crop(INPUT_IMAGE)

    # Step 2: initialize one pipeline and process only the crop.
    pipeline = initialize_pipeline()
    results = process_image(pipeline, crop_path)

    logger.info("Total results: %d", len(results))
    logger.info("Outputs saved under: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
