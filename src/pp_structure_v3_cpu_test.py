"""CPU-only PP-StructureV3 test for scanned English invoices.

Uses one PP-StructureV3 pipeline with:
- PP-OCRv6 small text detection and recognition
- PP-StructureV3 layout detection (installed default)
- Table classification
- Separate wired and wireless table-structure models
- Separate wired and wireless table-cell detection models

Run from the project's virtual environment. Model weights may be downloaded
on first initialization if they are not already cached locally.
"""

import logging
from pathlib import Path

from paddleocr import PPStructureV3


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use the same preprocessed invoice image as the earlier test.
INPUT_IMAGE = Path(
    r"C:\OCR_PROJECT\output\ocr_images\27th oct_page_1.png"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "pp_structure_v3_cpu_test"
JSON_DIR = OUTPUT_DIR / "json"
VISUAL_DIR = OUTPUT_DIR / "visualizations"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
HTML_DIR = OUTPUT_DIR / "html"
EXCEL_DIR = OUTPUT_DIR / "excel"

DEVICE = "cpu"

# OCR models: explicit names bypass PPStructureV3's ocr_version selector.
TEXT_DETECTION_MODEL = "PP-OCRv6_small_det"
TEXT_RECOGNITION_MODEL = "PP-OCRv6_small_rec"

# Table models available in the PaddleX 3.7 model registry.
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
# INITIALIZE ONE PP-STRUCTURE-V3 PIPELINE
# --------------------------------------------------

def initialize_pipeline():
    logger.info("Initializing PP-StructureV3 on CPU...")
    logger.info("OCR detection model: %s", TEXT_DETECTION_MODEL)
    logger.info("OCR recognition model: %s", TEXT_RECOGNITION_MODEL)
    logger.info("Wired table structure model: %s", WIRED_TABLE_STRUCTURE_MODEL)
    logger.info("Wireless table structure model: %s", WIRELESS_TABLE_STRUCTURE_MODEL)
    logger.info("Wired cell detection model: %s", WIRED_TABLE_CELLS_MODEL)
    logger.info("Wireless cell detection model: %s", WIRELESS_TABLE_CELLS_MODEL)

    pipeline = PPStructureV3(
        device=DEVICE,

        # Explicit OCR models. Do not pass ocr_version here: the installed
        # PPStructureV3 implementation only accepts v3/v4/v5 in that selector.
        text_detection_model_name=TEXT_DETECTION_MODEL,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL,
        text_recognition_batch_size=1,

        # Layout detection remains on the PP-StructureV3 installed default.
        # Table recognition is enabled with explicit wired/wireless models.
        use_table_recognition=True,
        table_classification_model_name=TABLE_CLASSIFICATION_MODEL,
        wired_table_structure_recognition_model_name=WIRED_TABLE_STRUCTURE_MODEL,
        wireless_table_structure_recognition_model_name=WIRELESS_TABLE_STRUCTURE_MODEL,
        wired_table_cells_detection_model_name=WIRED_TABLE_CELLS_MODEL,
        wireless_table_cells_detection_model_name=WIRELESS_TABLE_CELLS_MODEL,

        # The input image is already preprocessed; disable extra page modules.
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
# PROCESS ONE IMAGE
# --------------------------------------------------

def process_image(pipeline, image_path: Path):
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    logger.info("Processing one image: %s", image_path)
    results = pipeline.predict(input=str(image_path))

    if not results:
        raise RuntimeError("PP-StructureV3 returned no results.")

    for index, result in enumerate(results, start=1):
        logger.info("Saving result %s", index)

        result.save_to_json(save_path=str(JSON_DIR))
        result.save_to_img(save_path=str(VISUAL_DIR))
        result.save_to_markdown(save_path=str(MARKDOWN_DIR))

        # These exporters may not be available for every result/version.
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
    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(
            "Configure INPUT_IMAGE to point to an existing preprocessed invoice image. "
            f"Current value: {INPUT_IMAGE}"
        )

    for folder in (JSON_DIR, VISUAL_DIR, MARKDOWN_DIR, HTML_DIR, EXCEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # One pipeline initialization and one prediction call for this image.
    pipeline = initialize_pipeline()
    results = process_image(pipeline, INPUT_IMAGE)
    logger.info("Total results: %d", len(results))
    logger.info("Outputs saved under: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
