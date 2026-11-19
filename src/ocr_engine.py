"""PP-StructureV3 image-to-JSON engine for invoice images.

The PaddleOCR dependency is imported lazily so configuration validation and
unit tests can run without loading GPU libraries.
"""
from __future__ import annotations

import gc
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _to_jsonable(value: Any) -> Any:
    """Recursively convert Paddle/Numpy values to standard JSON values."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    return str(value)


def _find_key_recursive(data: Any, key: str) -> list[Any]:
    """Return every value stored under `key` in a nested dict/list structure."""
    found: list[Any] = []
    if isinstance(data, dict):
        for current_key, value in data.items():
            if current_key == key:
                found.append(value)
            found.extend(_find_key_recursive(value, key))
    elif isinstance(data, list):
        for value in data:
            found.extend(_find_key_recursive(value, key))
    return found


@dataclass(frozen=True)
class TableQuality:
    """Heuristic table-output assessment; it is not a ground-truth accuracy score."""

    table_count: int
    recognized_cell_count: int
    html_row_count: int
    html_cell_count: int
    mean_cell_confidence: float | None
    score: float
    is_poor: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "table_count": self.table_count,
            "recognized_cell_count": self.recognized_cell_count,
            "html_row_count": self.html_row_count,
            "html_cell_count": self.html_cell_count,
            "mean_cell_confidence": self.mean_cell_confidence,
            "heuristic_score": round(self.score, 4),
            "is_poor": self.is_poor,
            "reason": self.reason,
            "score_is_heuristic_not_model_accuracy": True,
        }


def evaluate_table_quality(
    result_payload: dict[str, Any],
    *,
    minimum_recognized_cells: int = 1,
    minimum_mean_confidence: float = 0.45,
) -> TableQuality:
    """Assess whether PP-StructureV3 produced minimally usable table output.

    The assessment uses available table HTML, cell OCR text and confidence
    values. It is intentionally conservative and should be calibrated against
    the project's invoice test set before production use.
    """
    table_lists = _find_key_recursive(result_payload, "table_res_list")
    tables: list[dict[str, Any]] = []
    for table_list in table_lists:
        if isinstance(table_list, list):
            tables.extend(item for item in table_list if isinstance(item, dict))

    if not tables:
        return TableQuality(
            table_count=0,
            recognized_cell_count=0,
            html_row_count=0,
            html_cell_count=0,
            mean_cell_confidence=None,
            score=0.0,
            is_poor=True,
            reason="No table regions/results were returned by PP-StructureV3.",
        )

    recognized_texts: list[str] = []
    confidence_values: list[float] = []
    row_count = 0
    html_cell_count = 0
    valid_html_tables = 0
    has_cell_geometry = False

    for table in tables:
        html = table.get("pred_html") or ""
        if isinstance(html, str):
            row_count += len(re.findall(r"<tr\b", html, flags=re.IGNORECASE))
            html_cell_count += len(re.findall(r"<(?:td|th)\b", html, flags=re.IGNORECASE))
            if "<table" in html.lower() and "</table>" in html.lower():
                valid_html_tables += 1

        boxes = table.get("cell_box_list") or []
        if isinstance(boxes, list) and boxes:
            has_cell_geometry = True

        table_ocr = table.get("table_ocr_pred") or {}
        if isinstance(table_ocr, dict):
            texts = table_ocr.get("rec_texts") or []
            scores = table_ocr.get("rec_scores") or []
            if isinstance(texts, list):
                recognized_texts.extend(
                    str(text).strip() for text in texts if str(text).strip()
                )
            if isinstance(scores, list):
                for score in scores:
                    try:
                        numeric_score = float(score)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= numeric_score <= 1.0:
                        confidence_values.append(numeric_score)

    recognized_count = len(recognized_texts)
    mean_confidence = mean(confidence_values) if confidence_values else None

    # Transparent weighted indicators, not a calibrated probability.
    html_component = 1.0 if valid_html_tables else 0.0
    structure_component = 1.0 if row_count > 0 and (html_cell_count > 0 or has_cell_geometry) else 0.0
    text_component = min(1.0, recognized_count / max(1, minimum_recognized_cells))
    confidence_component = (
        min(1.0, mean_confidence / max(minimum_mean_confidence, 1e-6))
        if mean_confidence is not None
        else 0.5
    )
    score = (
        0.30 * html_component
        + 0.25 * structure_component
        + 0.30 * text_component
        + 0.15 * confidence_component
    )

    reasons: list[str] = []
    if not valid_html_tables:
        reasons.append("table HTML missing or incomplete")
    if row_count == 0 or (html_cell_count == 0 and not has_cell_geometry):
        reasons.append("row/cell structure not found")
    if recognized_count < minimum_recognized_cells:
        reasons.append("too few recognized table-cell texts")
    if mean_confidence is not None and mean_confidence < minimum_mean_confidence:
        reasons.append("mean table-cell OCR confidence below threshold")

    return TableQuality(
        table_count=len(tables),
        recognized_cell_count=recognized_count,
        html_row_count=row_count,
        html_cell_count=html_cell_count,
        mean_cell_confidence=mean_confidence,
        score=score,
        is_poor=bool(reasons),
        reason="; ".join(reasons) if reasons else "Basic table-output checks passed.",
    )


class StructureEngine:
    """Owns one PP-StructureV3 pipeline and lazily creates a retry pipeline."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = dict(settings)
        self.device = str(self.settings.get("device", "gpu:0"))
        self.pipeline = self._create_pipeline(swapped_table_models=False)
        self._pipeline_mode = "primary"

    def _create_pipeline(self, *, swapped_table_models: bool) -> Any:
        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise ImportError(
                "PP-StructureV3 is unavailable. Install PaddleOCR 3.x with the "
                "document-parser dependencies and a compatible PaddlePaddle build."
            ) from exc

        table_cfg = self.settings.get("table", {})
        model_cfg = self.settings.get("models", {})
        layout_cfg = model_cfg.get("layout", {})

        wired_name = table_cfg.get("wired_model", "SLANeXt_wired")
        wireless_name = table_cfg.get("wireless_model", "SLANeXt_wireless")
        if swapped_table_models:
            wired_name, wireless_name = wireless_name, wired_name

        kwargs: dict[str, Any] = {
            "lang": self.settings.get("language", "en"),
            "device": self.device,
            "layout_detection_model_name": layout_cfg.get("model_name", "PP-DocLayout-L"),
            "text_detection_model_name": model_cfg.get("text_detection", {}).get(
                "model_name", "PP-OCRv5_server_det"
            ),
            "text_recognition_model_name": model_cfg.get("text_recognition", {}).get(
                "model_name", "PP-OCRv5_server_rec"
            ),
            "text_recognition_batch_size": int(self.settings.get("text_recognition_batch_size", 1)),
            "table_classification_model_name": table_cfg.get(
                "classification_model", "PP-LCNet_x1_0_table_cls"
            ),
            "wired_table_structure_recognition_model_name": wired_name,
            "wireless_table_structure_recognition_model_name": wireless_name,
            "use_doc_orientation_classify": bool(self.settings.get("use_doc_orientation_classify", False)),
            "use_doc_unwarping": bool(self.settings.get("use_doc_unwarping", False)),
            "use_textline_orientation": bool(self.settings.get("use_textline_orientation", False)),
            "use_seal_recognition": bool(self.settings.get("use_seal_recognition", False)),
            "use_table_recognition": bool(table_cfg.get("enabled", True)),
            "use_formula_recognition": bool(self.settings.get("use_formula_recognition", False)),
            "use_chart_recognition": bool(self.settings.get("use_chart_recognition", False)),
            "use_region_detection": bool(self.settings.get("use_region_detection", False)),
        }

        # Explicit model directories are optional. When omitted, PaddleX uses
        # its official model cache and downloads missing model weights.
        model_dir_settings = (
            (layout_cfg.get("model_dir"), "layout_detection_model_dir"),
            (model_cfg.get("text_detection", {}).get("model_dir"), "text_detection_model_dir"),
            (model_cfg.get("text_recognition", {}).get("model_dir"), "text_recognition_model_dir"),
            (table_cfg.get("table_classification_model_dir"), "table_classification_model_dir"),
            (table_cfg.get("wired_model_dir"), "wired_table_structure_recognition_model_dir"),
            (table_cfg.get("wireless_model_dir"), "wireless_table_structure_recognition_model_dir"),
        )
        for value, constructor_key in model_dir_settings:
            if value:
                kwargs[constructor_key] = str(value)

        logger.info(
            "Initializing PP-StructureV3 | device=%s | OCR=%s | layout=%s | wired=%s | wireless=%s | swapped=%s",
            self.device,
            kwargs["ocr_version"],
            kwargs["layout_detection_model_name"],
            wired_name,
            wireless_name,
            swapped_table_models,
        )
        return PPStructureV3(**kwargs)

    def _release_pipeline(self) -> None:
        self.pipeline = None
        gc.collect()
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("GPU cache cleanup was unavailable.", exc_info=True)

    def _ensure_primary(self) -> None:
        if self.pipeline is None:
            self.pipeline = self._create_pipeline(swapped_table_models=False)
            self._pipeline_mode = "primary"

    def _run_once(self, image_path: Path) -> dict[str, Any]:
        self._ensure_primary()
        predict_cfg = self.settings.get("prediction", {})
        results = self.pipeline.predict(
            input=str(image_path),
            use_doc_orientation_classify=bool(self.settings.get("use_doc_orientation_classify", False)),
            use_doc_unwarping=bool(self.settings.get("use_doc_unwarping", False)),
            use_textline_orientation=bool(self.settings.get("use_textline_orientation", False)),
            use_seal_recognition=bool(self.settings.get("use_seal_recognition", False)),
            use_table_recognition=bool(self.settings.get("table", {}).get("enabled", True)),
            use_formula_recognition=bool(self.settings.get("use_formula_recognition", False)),
            use_chart_recognition=bool(self.settings.get("use_chart_recognition", False)),
            use_region_detection=bool(self.settings.get("use_region_detection", False)),
            use_table_orientation_classify=bool(predict_cfg.get("use_table_orientation_classify", False)),
            use_ocr_results_with_table_cells=bool(predict_cfg.get("use_ocr_results_with_table_cells", True)),
        )
        results = list(results)
        if not results:
            raise RuntimeError(f"PP-StructureV3 returned no result for {image_path.name}.")
        if len(results) != 1:
            raise RuntimeError(
                f"Expected one result for image {image_path.name}, received {len(results)}."
            )
        payload = getattr(results[0], "json", None)
        if callable(payload):
            payload = payload()
        if payload is None:
            raise TypeError("PP-StructureV3 result does not expose the documented .json attribute.")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected result.json to be a dictionary, got {type(payload).__name__}.")
        return _to_jsonable(payload)

    def process_image(self, image_path: Path) -> tuple[dict[str, Any], bool]:
        """Run primary table mapping and, when indicated, retry swapped models."""
        image_path = Path(image_path)
        primary_payload = self._run_once(image_path)
        eval_cfg = self.settings.get("table_evaluation", {})
        primary_quality = evaluate_table_quality(
            primary_payload,
            minimum_recognized_cells=int(eval_cfg.get("minimum_recognized_cells", 1)),
            minimum_mean_confidence=float(eval_cfg.get("minimum_mean_confidence", 0.45)),
        )

        attempts = [{"model_mapping": "primary", **primary_quality.as_dict()}]
        selected_payload = primary_payload
        selected_quality = primary_quality
        retried = False

        should_retry = (
            bool(eval_cfg.get("enabled", True))
            and bool(eval_cfg.get("retry_with_alternative", True))
            and primary_quality.table_count > 0
            and primary_quality.is_poor
        )

        if should_retry:
            logger.warning(
                "Table output for %s did not pass basic checks (%s). Retrying with swapped SLANeXt mapping.",
                image_path.name,
                primary_quality.reason,
            )
            retried = True
            self._release_pipeline()
            try:
                self.pipeline = self._create_pipeline(swapped_table_models=True)
                self._pipeline_mode = "alternative"
                alternative_payload = self._run_once(image_path)
                alternative_quality = evaluate_table_quality(
                    alternative_payload,
                    minimum_recognized_cells=int(eval_cfg.get("minimum_recognized_cells", 1)),
                    minimum_mean_confidence=float(eval_cfg.get("minimum_mean_confidence", 0.45)),
                )
                attempts.append({"model_mapping": "alternative_swapped", **alternative_quality.as_dict()})
                if alternative_quality.score > primary_quality.score:
                    selected_payload = alternative_payload
                    selected_quality = alternative_quality
            except Exception as exc:
                # A failed retry must not discard the first-pass OCR result.
                attempts.append({
                    "model_mapping": "alternative_swapped",
                    "error": f"{type(exc).__name__}: {exc}",
                    "is_poor": True,
                })
                logger.exception("Alternative table-model retry failed for %s", image_path.name)
            finally:
                # Do not retain both model sets in VRAM. The primary pipeline
                # is re-created lazily before the next image.
                self._release_pipeline()

        manual_review = selected_quality.is_poor
        if selected_quality.table_count == 0:
            manual_review = bool(eval_cfg.get("review_when_no_table_detected", True))

        output = {
            "source_image": image_path.name,
            "pipeline": "PP-StructureV3",
            "configuration": {
                "language": self.settings.get("language", "en"),
                "ocr_version": self.settings.get("ocr_version", "PP-OCRv5"),
                "device": self.device,
                "layout_model": self.settings.get("models", {}).get("layout", {}).get("model_name", "PP-DocLayout-L"),
                "table_classifier": self.settings.get("table", {}).get("classification_model", "PP-LCNet_x1_0_table_cls"),
            },
            "table_evaluation": {
                "selection_method": "PP-StructureV3 table classification module",
                "attempts": attempts,
                "selected_model_mapping": attempts[-1]["model_mapping"] if selected_quality is not primary_quality else "primary",
                "selected_quality": selected_quality.as_dict(),
                "human_review_required": manual_review,
            },
            "paddle_result": selected_payload,
        }
        return output, retried


def initialize_ocr(
    language: str = "en",
    device: str = "gpu:0",
    settings: dict[str, Any] | None = None,
) -> StructureEngine:
    """Initialize PP-StructureV3 once and return a reusable engine.

    `language` and `device` retain compatibility with the previous main.py
    interface. Full settings from config.json can be supplied as `settings`.
    """
    effective_settings = dict(settings or {})
    effective_settings["language"] = language
    effective_settings["device"] = device
    if language != "en":
        raise ValueError("This invoice OCR configuration currently supports English only.")
    if not device.strip():
        raise ValueError("OCR device must be a non-empty string, e.g. 'gpu:0' or 'cpu'.")
    return StructureEngine(effective_settings)


def _atomic_write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(data), handle, ensure_ascii=False, indent=2)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_ocr_folder(
    input_folder: Path,
    output_folder: Path,
    ocr_engine: StructureEngine,
    recursive: bool = False,
    manual_review_folder: Path | None = None,
) -> dict[str, int]:
    """Process each image and save `<image_stem>.json` in the output folder."""
    input_folder = Path(input_folder).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve()
    review_folder = Path(manual_review_folder).expanduser().resolve() if manual_review_folder else None

    if not input_folder.is_dir():
        raise NotADirectoryError(f"OCR input folder does not exist: {input_folder}")
    if input_folder == output_folder:
        raise ValueError("OCR input and JSON output folders must be different.")

    output_folder.mkdir(parents=True, exist_ok=True)
    if review_folder:
        review_folder.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    image_paths = sorted(
        path for path in input_folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    summary = {
        "total_images": len(image_paths),
        "successful": 0,
        "failed": 0,
        "table_retries": 0,
        "manual_review_required": 0,
    }

    if not image_paths:
        logger.warning("No supported images found in %s", input_folder)
        return summary

    for image_path in image_paths:
        try:
            result, retried = ocr_engine.process_image(image_path)
            relative_parent = image_path.relative_to(input_folder).parent
            destination_folder = output_folder / relative_parent
            output_path = destination_folder / f"{image_path.stem}.json"
            _atomic_write_json(result, output_path)
            summary["successful"] += 1
            summary["table_retries"] += int(retried)

            if result.get("table_evaluation", {}).get("human_review_required", False):
                summary["manual_review_required"] += 1
                if review_folder:
                    review_destination = review_folder / relative_parent
                    review_destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, review_destination / image_path.name)
                    _atomic_write_json(
                        {
                            "source_image": image_path.name,
                            "json_result": str(output_path),
                            "reason": result.get("table_evaluation", {}).get("selected_quality", {}).get("reason"),
                            "status": "requires_human_review",
                        },
                        review_destination / f"{image_path.stem}.review.json",
                    )
            logger.info("Processed %s -> %s", image_path.name, output_path)

        except Exception:
            summary["failed"] += 1
            logger.exception("OCR processing failed for image: %s", image_path)

    logger.info("OCR batch summary: %s", summary)
    return summary
