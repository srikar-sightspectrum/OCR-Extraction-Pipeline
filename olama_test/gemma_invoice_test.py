
import argparse
import json
import os
import sys
import time

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemma4:e4b "

# If the script and Ollama run on the same machine, localhost works.
# Otherwise, set OLLAMA_URL to the remote server's IP address.

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://122.184.98.195:11434/api/generate"
)

NUM_CTX = 32768
NUM_PREDICT = 8192
TEMPERATURE = 0

REQUEST_TIMEOUT = 900

DEFAULT_OUTPUT = "gemma_invoice_result.json"


# ============================================================
# OUTPUT SCHEMA
# ============================================================

OUTPUT_SCHEMA = {
    "invoice_details": {
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "po_number": None,
        "gstin": None,
        "supplier_name": None,
        "buyer_name": None,
        "subtotal": None,
        "tax_amount": None,
        "discount": None,
        "total_amount": None,
        "currency": None,
        "payment_terms": None,
    },
    "line_items": [
        {
            "description": None,
            "hsn_sac": None,
            "quantity": None,
            "unit": None,
            "rate": None,
            "discount": None,
            "tax_rate": None,
            "tax_amount": None,
            "amount": None,
        }
    ],
}


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(page_data: dict) -> str:

    schema_text = json.dumps(
        OUTPUT_SCHEMA,
        ensure_ascii=False,
        separators=(",", ":")
    )

    ocr_text = json.dumps(
        page_data,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return f"""
You are a highly accurate invoice information extraction system.

TASK:
Extract invoice header fields and invoice line items from the supplied OCR JSON.

Use BOTH semantic understanding and spatial reasoning based on bounding_box coordinates to understand the document layout and identify the correct values.

GENERAL EXTRACTION RULES:

1. Return exactly one valid JSON object matching the output schema.
2. Do not include Markdown, explanations or text outside the JSON.
3. Use null when a field is absent or cannot be identified reliably.
4. Never invent, calculate or assume missing invoice information.
5. Preserve recognized values as text. Do not change date or amount formats.
6. gstin must contain the supplier/seller GSTIN, not the buyer GSTIN.
7. supplier_name is the seller/billed-from name.
8. buyer_name is the buyer/billed-to name.
9. Include only actual purchased goods or services in line_items.
10. Exclude headings, serial numbers, tax-summary rows, subtotal rows,
    CGST/SGST rows, grand-total rows and amount-in-words text.
11. Reconstruct descriptions split across nearby OCR regions.
12. Do not append serial numbers or quantity/unit text to descriptions.
13. Use coordinates to associate values with the correct row and column.
14. For subtotal, use the stated total taxable value if available.
15. For tax_amount, use the total tax only if explicitly stated.
    Do not calculate it from individual tax components.
16. For total_amount, use the numeric grand total.
17. Do not use the amount-in-words line as total_amount.
18. Keep line_items empty if no actual purchased items are present.
19. Do not confuse supplier information with buyer information.
20. Extract only information belonging to this invoice page.

SEMANTIC AND SPATIAL REASONING:

Use semantic understanding and coordinate relationships together.
Neither method should be used in isolation.

1. Understand the meaning and context of OCR text rather than relying
   only on exact keyword matching.

2. Use bounding_box coordinates to understand the relative positions
   of text regions.

3. Each bounding_box contains four corner coordinates:
   top-left, top-right, bottom-right and bottom-left.

4. The X-coordinate increases from left to right.
   The Y-coordinate increases from top to bottom.

5. Associate related text regions using their spatial proximity,
   horizontal alignment, vertical alignment and document structure.

6. Do not select a value based only on the nearest coordinates.
   Evaluate its semantic meaning and surrounding context as well.

INVOICE NUMBER EXTRACTION (HIGH PRIORITY):

The invoice number must be identified using BOTH semantic understanding
and coordinate-based spatial reasoning.

A. IDENTIFY THE HEADING:

Recognize invoice-number headings even when their wording varies.

Possible variations include, but are not limited to:

- Invoice Number
- Invoice No
- Invoice No.
- Invoice ID
- Invoice
- Inv No
- Inv. No.
- Inv Number
- Document Number

Do not depend on exact keyword matching.
Interpret the meaning of the text in its document context.

B. IDENTIFY THE ASSOCIATED VALUE:

After identifying a possible invoice-number heading:

1. Retrieve its bounding_box coordinates.
2. Examine the surrounding OCR text regions.
3. Search for a potential invoice number in the following positions:

   - To the right of the heading.
   - Directly below the heading.
   - Directly above the heading.
   - Within the same OCR text region as the heading.

4. Compare the candidate regions using horizontal and vertical
   alignment, relative distance and document layout.

5. Consider that the heading and value may be separated into
   different OCR text regions.

6. The value may contain letters, numbers, hyphens, slashes
   or other common invoice-number characters.

C. SEMANTIC VALIDATION:

For every potential candidate, evaluate:

- Whether its format and content are consistent with an invoice
  identifier.
- Whether its coordinates associate it with the invoice heading.
- Whether the surrounding text supports it being an invoice number.
- Whether it could instead represent a date, GSTIN, purchase order
  number, reference number or another document identifier.

Do not assume that every alphanumeric value is an invoice number.

D. COMBINED DECISION:

Select the candidate supported by the strongest combination of
semantic relevance, spatial association and document context.

If the heading and value appear in the same OCR text region,
extract only the value associated with the heading.

If the invoice number appears to the right, above or below the heading,
use the coordinates to establish the relationship.

Do not blindly select the closest text region.

Do not select a date, GSTIN, PO number or unrelated reference number
as the invoice number merely because it is spatially close.

If multiple candidates exist and the correct invoice number cannot
be identified reliably, return null.

Do not fabricate an invoice number when the heading or its associated
value cannot be identified.

HEADER AND LINE-ITEM ASSOCIATION:

1. Use semantic meaning to identify header labels and their associated
   values.
2. Use bounding_box coordinates to associate labels with nearby values.
3. Consider horizontal and vertical relationships when labels and
   values are separated.
4. For line items, identify column headings semantically and use their
   coordinates to establish column boundaries.
5. Associate values with the correct line-item row using vertical
   alignment and spatial proximity.
6. Do not associate values from adjacent rows or unrelated columns.
7. Reconstruct multi-region descriptions using their spatial order
   and semantic continuity.

LINE-ITEM SEQUENCE AND GAP VALIDATION (HIGH PRIORITY):

Ensure that ALL actual invoice line items are extracted, including
items appearing in the middle of the invoice table.

Do not stop extraction after identifying the first few line items.

A. IDENTIFY THE LINE-ITEM ROWS:

1. Identify the line-item table using semantic understanding of its
   column headings, such as:

   - S.No / Serial Number / Item No
   - Description / Particulars / Product / Service
   - HSN/SAC
   - Quantity
   - Rate / Unit Price
   - Amount / Taxable Value / Total

2. Use the bounding_box coordinates of the column headings to
   establish the table's column structure.

3. Identify every potential line-item row using semantic meaning,
   horizontal alignment and vertical positioning.

4. Group OCR text regions belonging to the same item into a single
   logical row.

5. Treat descriptions split across multiple OCR regions as part of
   the same item when their coordinates and semantic meaning support it.

B. CHECK VERTICAL ROW SPACING:

1. Determine the vertical position of each identified line-item row
   using the bounding_box coordinates of its associated OCR regions.

2. Arrange the identified line items in ascending vertical order,
   from top to bottom.

3. Compare the vertical gaps between consecutive line-item rows.

4. Examine whether the row spacing follows a reasonably consistent
   pattern across the invoice table.

5. If the gap between two consecutive identified line items is
   noticeably larger than the surrounding row gaps, treat it as a
   potential indication that an intermediate line item may have
   been missed.

6. When an irregular gap is detected, re-examine the OCR text regions
   located within that vertical interval.

7. Search for additional item descriptions, serial numbers, HSN/SAC
   values, quantities, rates and taxable values that may belong to
   a missing line item.

8. Use semantic understanding and horizontal column alignment to
   determine whether these regions form another actual line item.

C. MISSING LINE-ITEM RECOVERY:

1. Do not assume that a large vertical gap is automatically an empty
   space or a formatting difference.

2. Investigate every suspicious gap before finalizing extraction.

3. Check whether an intermediate serial number is present but its
   associated description or numerical values were overlooked.

4. Check whether a description exists without a clearly recognized
   serial number.

5. Check whether numerical values belonging to a potential item
   appear in the expected table columns within the suspicious gap.

6. If the OCR regions support an additional purchased item,
   reconstruct that item and include it in line_items.

7. Do not create a line item solely because a vertical gap is large.
   There must be supporting OCR text and semantic evidence.

D. FINAL SEQUENCE VALIDATION:

1. Re-examine the entire line-item table from top to bottom before
   returning the result.

2. Verify that every identifiable actual purchased item has been
   included.

3. Check the serial-number sequence when serial numbers are available.
   A missing or skipped serial number should trigger another examination
   of the corresponding vertical region.

4. Compare the vertical spacing between identified rows and investigate
   unusually large gaps.

5. Confirm that no identifiable intermediate line item has been
   overlooked.

6. Do not force uniform spacing when the invoice layout naturally
   contains different row heights, wrapped descriptions or section gaps.

7. Do not invent missing items or values to complete a sequence.

8. Preserve the original top-to-bottom order of the line items.

FINAL VALIDATION:

Before returning the JSON:

1. Verify that each extracted value belongs to the correct field.
2. Verify that the invoice number is associated with its heading
   using semantic and spatial evidence.
3. Check that dates, GSTINs and other identifiers have not been
   confused with the invoice number.
4. Check supplier and buyer information separately.
5. Check that line-item values belong to the correct rows and columns.
6. Verify that the entire line-item table has been examined.
7. Investigate unusually large vertical gaps between consecutive
   line items for potentially missed items.
8. Verify that the extracted line items follow the document's
   top-to-bottom order.
9. Ensure that missing or uncertain values are null rather than guessed.
10. Ensure the output strictly matches the supplied JSON schema.


OUTPUT SCHEMA:

{schema_text}

OCR JSON INPUT:

{ocr_text}

Return only valid JSON.
""".strip()


# ============================================================
# OLLAMA API
# ============================================================

def call_ollama(prompt: str) -> tuple[str, dict]:

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:

        with urlopen(
            request,
            timeout=REQUEST_TIMEOUT
        ) as response:

            raw_body = response.read().decode("utf-8")

    except HTTPError as exc:

        error_body = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Ollama HTTP {exc.code}: {error_body}"
        ) from exc

    except URLError as exc:

        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            f"Check server availability. Details: {exc}"
        ) from exc

    except TimeoutError as exc:

        raise RuntimeError(
            "Ollama request timed out."
        ) from exc

    # --------------------------------------------------------
    # Parse Ollama HTTP response
    # --------------------------------------------------------

    try:

        result = json.loads(raw_body)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Ollama returned invalid HTTP JSON.\n"
            f"Raw response: {raw_body[:3000]}"
        ) from exc

    if not isinstance(result, dict):

        raise RuntimeError(
            "Ollama HTTP response is not a JSON object."
        )

    # --------------------------------------------------------
    # Capture generation metadata
    # --------------------------------------------------------

    metadata = {
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "total_duration": result.get("total_duration"),
        "load_duration": result.get("load_duration"),
        "prompt_eval_duration": result.get("prompt_eval_duration"),
        "eval_duration": result.get("eval_duration"),
    }

    response_text = result.get("response")

    if not isinstance(response_text, str):

        raise RuntimeError(
            f"Ollama returned no valid response field: {result}"
        )

    return response_text, metadata


# ============================================================
# NORMALIZE AND VALIDATE MODEL OUTPUT
# ============================================================

def normalize_extraction(data: dict) -> dict:

    if not isinstance(data, dict):

        raise ValueError(
            "Model output must be a JSON object."
        )

    if not isinstance(data.get("invoice_details"), dict):

        raise ValueError(
            "invoice_details must be a JSON object."
        )

    if not isinstance(data.get("line_items"), list):

        raise ValueError(
            "line_items must be a JSON array."
        )

    # Preserve only schema fields and fill missing values with null.

    normalized = {
        "invoice_details": {},
        "line_items": [],
    }

    for key in OUTPUT_SCHEMA["invoice_details"]:

        normalized["invoice_details"][key] = (
            data["invoice_details"].get(key)
        )

    item_schema = OUTPUT_SCHEMA["line_items"][0]

    for index, item in enumerate(data["line_items"]):

        if not isinstance(item, dict):

            raise ValueError(
                f"Line item {index + 1} is not a JSON object."
            )

        normalized_item = {}

        for key in item_schema:

            normalized_item[key] = item.get(key)

        normalized["line_items"].append(normalized_item)

    return normalized


# ============================================================
# PROCESS ONE INVOICE PAGE
# ============================================================

def process_page(page_data: dict, page_index: int) -> dict:

    page_number = page_data.get(
        "page_number",
        page_index
    )

    print("\n" + "=" * 65)
    print(f"PROCESSING PAGE {page_number}")
    print("=" * 65)

    prompt = build_prompt(page_data)

    print(f"Prompt size: {len(prompt):,} characters")
    print(f"Context window: {NUM_CTX}")
    print(f"Maximum output tokens: {NUM_PREDICT}")

    print("\nSending request to Ollama...")

    start_time = time.time()

    response_text, metadata = call_ollama(prompt)

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Display Ollama metadata
    # --------------------------------------------------------

    print("\n--- OLLAMA GENERATION DETAILS ---")

    for key, value in metadata.items():

        print(f"{key}: {value}")

    print(f"Elapsed time: {elapsed:.2f} seconds")

    print("--- END DETAILS ---")

    # --------------------------------------------------------
    # Parse Gemma output
    # --------------------------------------------------------

    try:

        extracted = json.loads(response_text)

    except json.JSONDecodeError as exc:

        print("\nMODEL GENERATED INVALID JSON")

        print(f"Error: {exc}")

        print("\n--- RAW GEMMA RESPONSE ---")

        print(repr(response_text[:5000]))

        print("--- END RESPONSE ---")

        print("\nGeneration metadata:")

        print(json.dumps(
            metadata,
            indent=4
        ))

        if metadata.get("done_reason") == "length":

            print(
                "\nWARNING: Generation stopped because of "
                "a token limit."
            )

        elif metadata.get("eval_count") == 1:

            print(
                "\nWARNING: Only one token was generated. "
                "Check model availability and Ollama logs."
            )

        raise RuntimeError(
            f"Invalid model JSON on page {page_number}."
        ) from exc

    # --------------------------------------------------------
    # Normalize output
    # --------------------------------------------------------

    try:

        extracted = normalize_extraction(extracted)

    except ValueError as exc:

        raise RuntimeError(
            f"Invalid extraction structure on page "
            f"{page_number}: {exc}"
        ) from exc

    print(f"\nPage {page_number} extraction successful.")

    print(
        f"Line items found: {len(extracted['line_items'])}"
    )

    return {
        "page_number": page_number,
        "extraction": extracted,
        "generation_metadata": metadata,
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description="Gemma 4 E4B invoice extraction from OCR JSON."
    )

    parser.add_argument(
        "input_json",
        help="Path to combined OCR JSON."
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}"
    )

    args = parser.parse_args()

    input_path = Path(args.input_json).expanduser()

    output_path = Path(args.output).expanduser()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not input_path.is_file():

        print(
            f"ERROR: Input JSON not found: {input_path}",
            file=sys.stderr
        )

        return 1

    # --------------------------------------------------------
    # Load OCR JSON
    # --------------------------------------------------------

    try:

        with input_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            ocr_data = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:

        print(
            f"ERROR: Could not read OCR JSON: {exc}",
            file=sys.stderr
        )

        return 1

    if not isinstance(ocr_data, dict):

        print(
            "ERROR: OCR JSON root must be an object.",
            file=sys.stderr
        )

        return 1

    pages = ocr_data.get("pages")

    if not isinstance(pages, list) or not pages:

        print(
            "ERROR: OCR JSON must contain a non-empty pages array.",
            file=sys.stderr
        )

        return 1

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("GEMMA 4 E4B - INVOICE EXTRACTION")
    print("=" * 65)

    print(f"Input: {input_path.resolve()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Context: {NUM_CTX}")
    print(f"Maximum output: {NUM_PREDICT}")
    print(f"Total pages: {len(pages)}")

    print("\nEach invoice page will be processed independently.")

    # --------------------------------------------------------
    # Process pages independently
    # --------------------------------------------------------

    results = []

    for index, page_data in enumerate(pages, start=1):

        if not isinstance(page_data, dict):

            print(
                f"ERROR: Page {index} is not a JSON object.",
                file=sys.stderr
            )

            return 1

        try:

            page_result = process_page(
                page_data,
                index
            )

            results.append(page_result)

        except RuntimeError as exc:

            print(
                f"\nERROR: {exc}",
                file=sys.stderr
            )

            print(
                "\nProcessing stopped. "
                "No incomplete combined output will be saved."
            )

            return 1

    # --------------------------------------------------------
    # Build final combined output
    # --------------------------------------------------------

    final_output = {
        "source_file": input_path.name,
        "model": MODEL_NAME,
        "total_pages_processed": len(results),
        "pages": results,
    }

    # --------------------------------------------------------
    # Save final JSON
    # --------------------------------------------------------

    try:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                final_output,
                file,
                ensure_ascii=False,
                indent=4
            )

            file.write("\n")

    except OSError as exc:

        print(
            f"ERROR: Could not save output: {exc}",
            file=sys.stderr
        )

        return 1

    # --------------------------------------------------------
    # Save individual page results
    # --------------------------------------------------------

    for result in results:

        page_number = result["page_number"]

        individual_path = output_path.parent / (
            f"{output_path.stem}_page_{page_number}.json"
        )

        try:

            with individual_path.open(
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    result["extraction"],
                    file,
                    ensure_ascii=False,
                    indent=4
                )

                file.write("\n")

            print(
                f"Individual output saved: {individual_path}"
            )

        except OSError as exc:

            print(
                f"WARNING: Could not save page {page_number}: {exc}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ALL INVOICES PROCESSED SUCCESSFULLY")
    print("=" * 65)

    print(f"Pages processed: {len(results)}")

    print(f"Combined output: {output_path.resolve()}")

    print("\nExtraction summary:")

    for result in results:

        extraction = result["extraction"]

        details = extraction["invoice_details"]

        print(
            f"\nPage {result['page_number']}: "
            f"Invoice {details['invoice_number']}"
        )

        print(
            f"Line items: {len(extraction['line_items'])}"
        )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(main())
