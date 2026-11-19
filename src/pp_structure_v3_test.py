
import logging
from pathlib import Path

from paddleocr import PPStructureV3


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use the SAME preprocessed image that generated
# your existing OCR JSON.
INPUT_IMAGE = Path(
    r"C:\OCR_PROJECT\output\ocr_images\27th oct_page_1.png"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "pp_structure_v3_test"

JSON_DIR = OUTPUT_DIR / "json"
VISUAL_DIR = OUTPUT_DIR / "visualizations"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
HTML_DIR = OUTPUT_DIR / "html"
EXCEL_DIR = OUTPUT_DIR / "excel"

DEVICE = "cpu"  # Change to "gpu:0" when GPU setup is verified


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# INITIALIZE PP-STRUCTURE-V3
# --------------------------------------------------

def initialize_pipeline():

    logger.info("Initializing PP-StructureV3...")

    pipeline = PPStructureV3(
        lang="en",

        device=DEVICE,

        # We already preprocess the images.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # Required for our experiment.
        use_table_recognition=True,

        # Not required for this invoice experiment.
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )

    logger.info("PP-StructureV3 initialized.")

    return pipeline


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

def process_image(pipeline, image_path):

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    logger.info("Processing: %s", image_path.name)

    results = pipeline.predict(input=str(image_path))

    if not results:
        raise RuntimeError("PP-StructureV3 returned no results.")

    for index, result in enumerate(results, start=1):

        logger.info("Saving result %s", index)

        # Complete structured prediction JSON
        result.save_to_json(
            save_path=str(JSON_DIR)
        )

        # Intermediate visualizations
        result.save_to_img(
            save_path=str(VISUAL_DIR)
        )

        # Markdown representation
        result.save_to_markdown(
            save_path=str(MARKDOWN_DIR)
        )

        # Table HTML, if supported by installed version
        if hasattr(result, "save_to_html"):
            result.save_to_html(
                save_path=str(HTML_DIR)
            )

        # Table Excel output, if supported
        if hasattr(result, "save_to_xlsx"):
            result.save_to_xlsx(
                save_path=str(EXCEL_DIR)
            )

        logger.info("Result %s saved.", index)

    logger.info(
        "PP-StructureV3 processing completed successfully."
    )

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(
            f"Please configure INPUT_IMAGE correctly: {INPUT_IMAGE}"
        )

    for folder in [
        JSON_DIR,
        VISUAL_DIR,
        MARKDOWN_DIR,
        HTML_DIR,
        EXCEL_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    pipeline = initialize_pipeline()

    results = process_image(
        pipeline,
        INPUT_IMAGE
    )

    logger.info("Total results: %s", len(results))


if __name__ == "__main__":

    main()
