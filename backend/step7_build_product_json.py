"""
STEP 7 (master plan Step 16): Build structured product.json from OCR text.

Why we do this:
Step 6 gave us raw OCR text from a product label. Now we need to pull
specific fields out of that text (net quantity, MRP, manufacturer,
dates, batch number, etc.) into the structured product JSON shape
the compliance engine expects.

IMPORTANT (per project rules):
  - If a field can't be reliably found, it stays EMPTY. We never invent
    a value just to fill the field.
  - This uses simple regex pattern matching, not an LLM. It's
    deterministic and explainable, even if it's not perfect. Patterns
    can (and will) need tuning as we test against real label photos
    with different phrasing.
  - We attach a rough per-field confidence based on whether a pattern
    matched cleanly, so downstream steps know how much to trust each
    field.

Input : ocr_text/<name>.txt
Output: dataset/product.json
"""

import os
import re
import json

OCR_TEXT_PATH = "ocr_text/sample_product_label.txt"
OUTPUT_JSON = "dataset/product.json"

# -----------------------------------------------------------------
# FIELD EXTRACTION PATTERNS
# -----------------------------------------------------------------
# Each pattern is intentionally simple and specific to common label
# phrasing. Add more alternate phrasings here as you test real labels
# (e.g. "Manufactured By", "Mfd By", "Mkt By" all mean the same thing).
FIELD_PATTERNS = {
    "net_quantity": r"Net\s*Quantity\s*[:\-]?\s*([\d.]+\s*\w*)",
    "mrp": r"MRP\s*[:\-]?\s*(?:Rs\.?|₹|INR)?\s*([\d.,]+)",
    "batch_number": r"Batch\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)",
    "manufacturing_date": r"Mfg\.?\s*Date\s*[:\-]?\s*([\d/\-]+)",
    "expiry_date": r"Exp\.?\s*Date\s*[:\-]?\s*([\d/\-]+)",
    "consumer_care_phone": r"Consumer\s*Care\s*[:\-]?\s*([\d\-]+)",
    "consumer_care_email": r"([\w.\-]+@[\w.\-]+\.\w+)",
    # NOTE: "ingredients" removed from here — real ingredient lists wrap
    # across multiple OCR lines, so it needs the multi-line block
    # extraction below, not a single-line regex (see extract_ingredients_block).
}

# Manufacturer name/address needs special handling: it's usually a
# block of 1-3 lines AFTER a phrase like "Manufactured by:". We look
# for that anchor line, then take the following non-empty lines until
# a blank line or another known field label starts.
MANUFACTURER_ANCHOR_PATTERN = re.compile(
    r"(Manufactured\s*by|Mfd\.?\s*by|Mkt\.?\s*by|Packed\s*by)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Lines that indicate we've moved on to a different field (so
# manufacturer block extraction knows where to stop).
STOP_LINE_PATTERN = re.compile(
    r"^(Batch|Mfg|Exp|Best\s*Before|Net\s*Quantity|MRP|Ingredients|Consumer\s*Care|"
    r"CONTAINS|ALLERGEN|MANUFACTURED|PROCESSED\s+ON)",
    re.IGNORECASE,
)


def extract_ingredients_block(text: str):
    """
    Real ingredient lists commonly wrap across MULTIPLE OCR lines (one
    long comma-separated list). This finds the 'Ingredients:' anchor
    line, then keeps appending following lines until a blank line or a
    known stop-word line (e.g. 'CONTAINS:') is hit — same block-capture
    technique used for the manufacturer address below.
    """
    lines = text.splitlines()
    parts = []
    found_anchor = False

    for i, line in enumerate(lines):
        anchor_match = re.match(r"\s*Ingredients\s*[:\-]?\s*(.*)", line, re.IGNORECASE)
        if anchor_match:
            found_anchor = True
            first_part = anchor_match.group(1).strip()
            if first_part:
                parts.append(first_part)

            for next_line in lines[i + 1:]:
                stripped = next_line.strip()
                if stripped == "" or STOP_LINE_PATTERN.match(stripped):
                    break
                parts.append(stripped)
            break

    confidence = "matched" if found_anchor and parts else "not_found"
    return (" ".join(parts).strip() if parts else ""), confidence


def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run step6_ocr_product_image.py first, "
            f"or update OCR_TEXT_PATH to point at your OCR'd product text."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_simple_fields(text: str):
    """
    Runs each regex against the text ONE LINE AT A TIME (not the whole
    blob). This avoids a common bug where '\\s*' in a pattern accidentally
    matches across a newline and swallows part of the NEXT line's label
    into the current field's value.
    """
    lines = text.splitlines()
    extracted = {}
    confidence = {}

    for field, pattern in FIELD_PATTERNS.items():
        found_value = None
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                found_value = match.group(1).strip().rstrip(",")
                break
        extracted[field] = found_value if found_value else ""
        confidence[field] = "matched" if found_value else "not_found"

    return extracted, confidence


def extract_manufacturer_block(text: str):
    """
    Finds a line like 'Manufactured by:' and takes the following lines
    as name (first line) + address (remaining lines) until a blank
    line or a known field label appears.
    """
    lines = text.splitlines()
    name = ""
    address_lines = []
    found_anchor = False

    for i, line in enumerate(lines):
        if MANUFACTURER_ANCHOR_PATTERN.search(line.strip()):
            found_anchor = True
            # collect the following lines
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line == "" or STOP_LINE_PATTERN.match(next_line):
                    break
                if name == "":
                    name = next_line.rstrip(",")
                else:
                    address_lines.append(next_line.rstrip(","))
            break

    address = ", ".join(address_lines)
    confidence = "matched" if found_anchor and name else "not_found"
    return name, address, confidence


def split_quantity_and_unit(net_quantity_raw: str):
    """'60 g' -> ('60 g', 'g'). If no unit found, unit stays empty."""
    match = re.match(r"([\d.]+)\s*([a-zA-Z]*)", net_quantity_raw)
    if not match:
        return net_quantity_raw, ""
    number, unit = match.group(1), match.group(2)
    return net_quantity_raw, unit


def build_product_json(text: str, ocr_source: str):
    simple_fields, simple_confidence = extract_simple_fields(text)
    manufacturer_name, manufacturer_address, mfr_confidence = extract_manufacturer_block(text)
    ingredients, ingredients_confidence = extract_ingredients_block(text)

    net_quantity_raw = simple_fields.pop("net_quantity")
    net_quantity, quantity_unit = split_quantity_and_unit(net_quantity_raw)

    product = {
        "product_name": "",  # not reliably auto-detectable yet; left for manual entry or later heuristic
        "common_or_generic_name": "",
        "manufacturer_name": manufacturer_name,
        "manufacturer_address": manufacturer_address,
        "packer_name": "",
        "importer_name": "",
        "country_of_origin": "",
        "net_quantity": net_quantity,
        "quantity_unit": quantity_unit,
        "mrp": simple_fields["mrp"],
        "manufacturing_date": simple_fields["manufacturing_date"],
        "expiry_date": simple_fields["expiry_date"],
        "best_before": "",
        "batch_number": simple_fields["batch_number"],
        "consumer_care_email": simple_fields["consumer_care_email"],
        "consumer_care_phone": simple_fields["consumer_care_phone"],
        "dimensions": "",
        "ingredients": ingredients,
        "nutrition": {},
        "claims": [],
        "directions": "",
        "storage": "",
        "ocr_confidence": {
            **simple_confidence,
            "manufacturer_name": mfr_confidence,
            "manufacturer_address": mfr_confidence,
            "ingredients": ingredients_confidence,
        },
        "ocr_source": ocr_source,
    }
    return product


if __name__ == "__main__":
    text = load_text(OCR_TEXT_PATH)
    print(f"Loaded OCR text from '{OCR_TEXT_PATH}' ({len(text)} characters)\n")

    product = build_product_json(text, ocr_source=OCR_TEXT_PATH)

    print("Extracted fields:")
    for key, value in product.items():
        if key in ("nutrition", "claims", "ocr_confidence"):
            continue
        marker = "  " if str(value).strip() else "  [EMPTY] "
        print(f"{marker}{key}: {value!r}")

    not_found = [f for f, c in product["ocr_confidence"].items() if c == "not_found"]
    if not_found:
        print(f"\nNOTE: these fields could not be found by the current regex "
              f"patterns and were left empty (not invented): {not_found}")
        print("If the real label uses different wording for these, we can add "
              "alternate patterns to FIELD_PATTERNS.")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(product, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_JSON}")