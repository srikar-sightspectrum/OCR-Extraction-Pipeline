"""
Combine page-level OCR JSON files by PDF name and page number.

Expected input filenames:
    sample invoice_page_1.json
    sample invoice_page_2.json
    invoice_002_page_1.json
    invoice_002_page_2.json

The script scans one input folder, groups files by the name before
"_page_<number>", sorts pages numerically, and writes one JSON per PDF
into the output folder.

Only files matching "*_page_<number>.json" are processed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PAGE_FILENAME_PATTERN = re.compile(
    r"^(?P<document_name>.+)_page_(?P<page_number>\d+)$",
    flags=re.IGNORECASE,
)


def load_page_file(json_path: Path) -> dict[str, Any]:
    """Load a page OCR JSON and validate the fields needed for combining."""
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")

    pages = data.get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise ValueError(
            "Expected exactly one page in this page-level OCR JSON."
        )

    page = pages[0]
    if not isinstance(page, dict):
        raise ValueError("The page entry must be an object.")

    text_regions = page.get("text_regions")
    if not isinstance(text_regions, list):
        raise ValueError("'text_regions' must be a list.")

    for index, region in enumerate(text_regions, start=1):
        if not isinstance(region, dict):
            raise ValueError(f"Text region {index} must be an object.")
        if not isinstance(region.get("text"), str):
            raise ValueError(f"Text region {index} has invalid or missing text.")
        if "confidence" not in region:
            raise ValueError(f"Text region {index} is missing confidence.")
        if "bounding_box" not in region:
            raise ValueError(f"Text region {index} is missing bounding_box.")

    return data


def combine_folder(
    input_folder: Path,
    output_folder: Path,
) -> list[Path]:
    """
    Scan input_folder, group page JSON files by document name,
    and write one combined JSON per document.
    """
    input_folder = input_folder.expanduser().resolve()
    output_folder = output_folder.expanduser().resolve()

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")

    if input_folder == output_folder:
        raise ValueError("Input and output folders must be different.")

    output_folder.mkdir(parents=True, exist_ok=True)

    grouped_files: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    skipped_files: list[str] = []

    # Scan only JSON files directly inside the supplied folder.
    for json_path in input_folder.glob("*.json"):
        match = PAGE_FILENAME_PATTERN.match(json_path.stem)

        if not match:
            skipped_files.append(json_path.name)
            continue

        document_name = match.group("document_name")
        page_number = int(match.group("page_number"))

        grouped_files[document_name].append((page_number, json_path))

    if not grouped_files:
        raise FileNotFoundError(
            "No matching files found. Expected filenames such as "
            "'sample invoice_page_1.json'."
        )

    created_files: list[Path] = []

    for document_name, page_files in sorted(
        grouped_files.items(),
        key=lambda item: item[0].lower(),
    ):
        page_files.sort(key=lambda item: item[0])

        page_numbers = [number for number, _ in page_files]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError(
                f"Duplicate page number found for document '{document_name}'."
            )

        combined_pages: list[dict[str, Any]] = []
        total_text_regions = 0

        for page_number, json_path in page_files:
            page_data = load_page_file(json_path)
            page = dict(page_data["pages"][0])

            # The filename is the source of truth for the document-level page number.
            page["page_number"] = page_number

            combined_pages.append(page)
            total_text_regions += len(page["text_regions"])

        combined_data = {
            "document_name": document_name,
            "total_text_regions": total_text_regions,
            "pages": combined_pages,
        }

        output_path = output_folder / f"{document_name}.json"
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(combined_data, file, ensure_ascii=False, indent=4)
                file.write("\n")
            temporary_path.replace(output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        created_files.append(output_path)

        print(
            f"Created: {output_path.name} | "
            f"Pages: {len(combined_pages)} | "
            f"Text regions: {total_text_regions}"
        )

    if skipped_files:
        print("\nSkipped files that did not match '*_page_<number>.json':")
        for filename in skipped_files:
            print(f"  - {filename}")

    return created_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine page-level OCR JSON files into one JSON per PDF, "
            "grouped by filename and page number."
        )
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing all page-level OCR JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="combined_json",
        help="Folder for combined JSON files (default: combined_json).",
    )

    args = parser.parse_args()

    try:
        outputs = combine_folder(
            input_folder=Path(args.input_folder),
            output_folder=Path(args.output_folder),
        )
    except (OSError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        return 1

    print(f"\nCompleted. Combined {len(outputs)} PDF document(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
