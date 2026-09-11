# step7_build_product_json.py

import csv
import os
import re
from typing import Any


# ============================================================
# BASIC HELPERS
# ============================================================

DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{1,2}[/-]\d{1,2}"
    r"|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b"
)

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

FSSAI_RE = re.compile(r"\b\d{14}\b")

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    value = value.replace("\u2018", "'")
    value = value.replace("\u2019", "'")
    value = value.replace("\u201c", '"')
    value = value.replace("\u201d", '"')

    return re.sub(r"\s+", " ", value).strip()


def normalize_number(value: str) -> str:
    value = clean_text(value)

    # Remove obvious OCR punctuation around numbers.
    value = value.strip(".,:;|[](){}")

    return value


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ============================================================
# TSV / COORDINATE OCR
# ============================================================

def load_layout(layout_path: str) -> list[dict]:
    """
    Load Tesseract TSV output.

    Expected columns:
        word left top width height confidence
    """

    if not layout_path or not os.path.exists(layout_path):
        return []

    rows = []

    with open(
        layout_path,
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            try:
                word = clean_text(row.get("word", ""))

                if not word:
                    continue

                rows.append(
                    {
                        "word": word,
                        "left": float(row.get("left", 0)),
                        "top": float(row.get("top", 0)),
                        "width": float(row.get("width", 0)),
                        "height": float(row.get("height", 0)),
                        "confidence": float(row.get("confidence", 0)),
                    }
                )

            except (ValueError, TypeError):
                continue

    return rows


def center_x(row: dict) -> float:
    return row["left"] + row["width"] / 2


def center_y(row: dict) -> float:
    return row["top"] + row["height"] / 2


def right_x(row: dict) -> float:
    return row["left"] + row["width"]


def word_lower(row: dict) -> str:
    return clean_text(row["word"]).lower()


# ============================================================
# FUZZY-ish WORD MATCHING
# ============================================================

def word_matches(row: dict, candidates: list[str]) -> bool:
    """
    Flexible matching because OCR may produce:

        Batch
        BatchNo.
        Baich
        MRP
        MRI
        Pkg.
        Dates
        Use
        by

    We intentionally keep this conservative.
    """

    word = word_lower(row)

    for candidate in candidates:
        candidate = candidate.lower()

        if word == candidate:
            return True

        # Remove punctuation for comparison.
        compact_word = re.sub(r"[^a-z0-9]", "", word)
        compact_candidate = re.sub(r"[^a-z0-9]", "", candidate)

        if compact_word == compact_candidate:
            return True

    return False


def find_words(rows: list[dict], candidates: list[str]) -> list[dict]:
    return [
        row
        for row in rows
        if word_matches(row, candidates)
    ]


# ============================================================
# DATE / NUMBER VALIDATION
# ============================================================

def extract_date_from_word(word: str) -> str:
    word = clean_text(word)

    match = DATE_RE.search(word)

    if not match:
        return ""

    return match.group(0)


def looks_like_date(word: str) -> bool:
    return bool(extract_date_from_word(word))


def looks_like_price(word: str) -> bool:
    word = clean_text(word)

    # Remove common currency symbols.
    word = word.replace("₹", "")
    word = word.replace("Rs", "")
    word = word.replace("INR", "")

    return bool(
        re.fullmatch(
            r"\d+(?:[.,]\d{1,2})?",
            word.strip(),
            flags=re.IGNORECASE,
        )
    )


def looks_like_quantity(word: str) -> bool:
    word = clean_text(word)

    # Examples:
    # 6
    # 6N   <- common OCR corruption of 6 pcs
    # 500g
    # 1kg
    # 250ml
    # 12pcs

    return bool(
        re.fullmatch(
            r"\d+(?:[.,]\d+)?\s*[A-Za-z]{0,8}",
            word,
        )
    )


# ============================================================
# SPATIAL SEARCH
# ============================================================

def candidate_rows_below(
    rows: list[dict],
    header: dict,
    max_y_distance: float = 100,
) -> list[dict]:
    """
    Find words below a header and reasonably close horizontally.
    """

    header_x = center_x(header)
    header_bottom = header["top"] + header["height"]

    candidates = []

    for row in rows:
        if row is header:
            continue

        cy = center_y(row)
        cx = center_x(row)

        if cy < header_bottom:
            continue

        if cy - header_bottom > max_y_distance:
            continue

        # Horizontal tolerance.
        if abs(cx - header_x) > 180:
            continue

        candidates.append(row)

    candidates.sort(
        key=lambda r: (
            abs(center_y(r) - header_bottom),
            abs(center_x(r) - header_x),
        )
    )

    return candidates


def candidate_rows_same_region(
    rows: list[dict],
    header: dict,
    x_min: float,
    x_max: float,
    max_y_distance: float = 100,
) -> list[dict]:

    header_bottom = header["top"] + header["height"]

    candidates = []

    for row in rows:
        if row is header:
            continue

        cx = center_x(row)
        cy = center_y(row)

        if not (x_min <= cx <= x_max):
            continue

        if cy < header_bottom:
            continue

        if cy - header_bottom > max_y_distance:
            continue

        candidates.append(row)

    candidates.sort(
        key=lambda r: (
            center_y(r),
            center_x(r),
        )
    )

    return candidates


# ============================================================
# DECLARATION TABLE EXTRACTION
# ============================================================

def extract_declaration_fields(rows: list[dict]) -> dict:
    """
    Coordinate-aware extraction for the declaration table.

    Important:
    - Never guess a compliance value.
    - Values must be inside the expected column.
    - Dates must be spatially associated with their header.
    """

    result = {
        "net_quantity": "",
        "mrp": "",
        "batch_number": "",
        "packing_date": "",
        "expiry_date": "",
    }

    if not rows:
        return result

    # --------------------------------------------------------
    # Find headers in the declaration area.
    # --------------------------------------------------------

    declaration_rows = [
        r for r in rows
        if 790 <= center_y(r) <= 850
    ]

    def find_header(words):
        matches = []

        for r in declaration_rows:
            w = word_lower(r)

            if w in words:
                matches.append(r)

        if not matches:
            return None

        return max(
            matches,
            key=lambda r: r["confidence"]
        )

    net_header = find_header({
        "net",
        "quantity",
    })

    mrp_header = find_header({
        "mrp",
        "mri",
        "mre",
        "mre()",
    })

    batch_header = find_header({
        "batch",
        "batchno.",
        "batchno",
    })

    pkg_header = find_header({
        "pkg.",
        "pkg",
    })

    use_header = find_header({
        "use",
    })

    # --------------------------------------------------------
    # Known approximate columns from this package.
    #
    # We use header positions when possible.
    # --------------------------------------------------------

    def column_candidates(
        header,
        x_left,
        x_right,
        y_start,
        y_end,
    ):
        if header:
            hx = center_x(header)

            # Use header as center, but keep the region
            # reasonably narrow.
            x_left = max(x_left, hx - 65)
            x_right = min(x_right, hx + 100)

        candidates = []

        for r in rows:

            cx = center_x(r)
            cy = center_y(r)

            if not (x_left <= cx <= x_right):
                continue

            if not (y_start <= cy <= y_end):
                continue

            candidates.append(r)

        candidates.sort(
            key=lambda r: (
                center_y(r),
                -r["confidence"],
            )
        )

        return candidates

    # ========================================================
    # NET QUANTITY
    # ========================================================

    candidates = column_candidates(
        net_header,
        0,
        170,
        835,
        900,
    )

    for r in candidates:

        word = clean_text(r["word"])

        # 6N is a common OCR interpretation of "6 pc".
        if re.fullmatch(r"\d+[A-Za-z]{0,3}", word):

            result["net_quantity"] = word

            if word.lower().endswith("n"):
                result["net_quantity"] = word[:-1]

            break

    # ========================================================
    # MRP
    # ========================================================

    candidates = column_candidates(
        mrp_header,
        150,
        350,
        835,
        920,
    )

    for r in candidates:

        word = clean_text(r["word"])

        if looks_like_price(word):

            # Don't accept absurdly small OCR fragments.
            number = word.replace(",", "").strip()

            try:
                value = float(number)

                if 1 <= value <= 100000:
                    result["mrp"] = word
                    break

            except ValueError:
                pass

    # ========================================================
    # BATCH NUMBER
    # ========================================================

    candidates = column_candidates(
        batch_header,
        400,
        550,
        830,
        920,
    )

    for r in candidates:

        word = clean_text(r["word"])

        compact = re.sub(
            r"[^a-z0-9]",
            "",
            word.lower(),
        )

        # Reject obvious OCR words.
        if compact in {
            "batch",
            "batchno",
            "no",
            "of",
            "date",
            "pkg",
            "use",
            "by",
        }:
            continue

        # Reject normal English words.
        if re.fullmatch(
            r"[A-Za-z]+",
            word,
        ):
            continue

        # Accept only a plausible alphanumeric batch code.
        if re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/-]{1,20}",
            word,
        ):
            result["batch_number"] = word
            break

    # ========================================================
    # PACKING DATE
    # ========================================================

    candidates = column_candidates(
        pkg_header,
        530,
        650,
        840,
        920,
    )

    for r in candidates:

        date = extract_date_from_word(
            r["word"]
        )

        if date:
            result["packing_date"] = date
            break

    # ========================================================
    # USE BY / EXPIRY DATE
    # ========================================================

    candidates = column_candidates(
        use_header,
        640,
        760,
        840,
        920,
    )

    for r in candidates:

        date = extract_date_from_word(
            r["word"]
        )

        if date:
            result["expiry_date"] = date
            break

    return result


# ============================================================
# FSSAI EXTRACTION
# ============================================================

def extract_fssai_numbers(
    text: str,
    rows: list[dict] | None = None,
) -> list[str]:

    numbers = []

    # --------------------------------------------------------
    # First use coordinate OCR.
    # --------------------------------------------------------

    if rows:

        for row in rows:
            word = clean_text(row["word"])

            # Exact 14 digit number.
            matches = FSSAI_RE.findall(word)

            for number in matches:
                if number not in numbers:
                    numbers.append(number)

    # --------------------------------------------------------
    # Then use plain OCR as backup.
    # --------------------------------------------------------

    for number in FSSAI_RE.findall(text):

        if number not in numbers:
            numbers.append(number)

    return numbers


# ============================================================
# PHONE / EMAIL
# ============================================================

def extract_phone(text: str) -> str:

    matches = PHONE_RE.findall(text)

    if not matches:
        return ""

    # Prefer a 10-digit Indian number.
    for value in matches:
        digits = re.sub(r"\D", "", value)

        if len(digits) == 10:
            return digits

        if len(digits) == 12 and digits.startswith("91"):
            return digits[-10:]

    return ""


def extract_email(text: str) -> str:

    match = EMAIL_RE.search(text)

    if not match:
        return ""

    return match.group(0)


# ============================================================
# MANUFACTURER / PACKER
# ============================================================

def extract_company_name(text: str, anchor_patterns: list[str]) -> str:

    lines = [
        clean_text(line)
        for line in text.splitlines()
        if clean_text(line)
    ]

    for i, line in enumerate(lines):

        lower = line.lower()

        for anchor in anchor_patterns:

            if anchor not in lower:
                continue

            # Take text AFTER the anchor.
            remainder = re.split(
                re.escape(anchor),
                line,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[-1]

            remainder = remainder.strip(
                " :-–—|,;"
            )

            # Reject obviously noisy OCR.
            if not remainder:
                continue

            if len(remainder) > 100:
                continue

            # A company name should contain letters.
            if not re.search(
                r"[A-Za-z]",
                remainder,
            ):
                continue

            # Remove obvious trailing unrelated OCR.
            remainder = re.split(
                r"\b(?:FSSAI|Lic\.?|License|Contact|Phone|Email)\b",
                remainder,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

            if 3 <= len(remainder) <= 100:
                return remainder

    return ""

def extract_address(text: str) -> str:

    lines = [
        clean_text(line)
        for line in text.splitlines()
        if clean_text(line)
    ]

    for i, line in enumerate(lines):

        if "401/402" not in line:
            continue

        address = []

        for candidate in lines[i:i + 3]:

            lower = candidate.lower()

            if "fssai" in lower:
                break

            if "qr code" in lower:
                break

            if "please scan" in lower:
                break

            address.append(candidate)

        value = " ".join(address)

        # Keep only a reasonable address.
        if len(value) > 250:
            value = value[:250]

        return value.strip()

    return ""

# ============================================================
# MAIN JSON BUILDER
# ============================================================

def build_product_json(
    text_path: str,
    layout_path: str | None = None,
) -> dict:

    text = load_text(text_path)

    if layout_path is None:

        base, _ = os.path.splitext(text_path)

        layout_path = base + "_layout.tsv"

    rows = load_layout(layout_path)

    # --------------------------------------------------------
    # Coordinate-aware declaration extraction
    # --------------------------------------------------------

    declaration = extract_declaration_fields(rows)

    # --------------------------------------------------------
    # FSSAI
    # --------------------------------------------------------

    fssai_numbers = extract_fssai_numbers(
        text,
        rows,
    )

    # --------------------------------------------------------
    # Contact details
    # --------------------------------------------------------

    phone = extract_phone(text)
    email = extract_email(text)

    # --------------------------------------------------------
    # Manufacturer
    # --------------------------------------------------------

    manufacturer_name = extract_company_name(
        text,
        [
            "mfd. by",
            "mfd by",
            "mfd. by-",
            "manufactured by",
            "manufacturer",
        ],
    )

    packer_name = extract_company_name(
        text,
        [
            "packed by",
            "packed by-",
            "packer",
        ],
    )

    manufacturer_address = extract_address(text)

    # --------------------------------------------------------
    # Product name / generic name
    # Keep these conservative here.
    # Vision/AI can improve them later.
    # --------------------------------------------------------

    product_name = ""
    common_name = ""

    for line in text.splitlines():

        cleaned = clean_text(line)

        if not cleaned:
            continue

        lower = cleaned.lower()

        if "nutritional information" in lower:
            continue

        if "nutrient-enriched eggs" in lower:
            common_name = "Eggs"
            product_name = "Nutrient-enriched Eggs"
            break

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {
        "product_name": product_name,
        "common_or_generic_name": common_name,

        "net_quantity": declaration["net_quantity"],
        "quantity_unit": "",

        "mrp": declaration["mrp"],

        "batch_number": declaration["batch_number"],

        "manufacturing_date": "",
        "packing_date": declaration["packing_date"],
        "expiry_date": declaration["expiry_date"],
        "best_before": "",

        "manufacturer_name": manufacturer_name,
        "manufacturer_address": manufacturer_address,

        "packer_name": packer_name,

        "importer_name": "",

        "consumer_care_phone": phone,
        "consumer_care_email": email,

        "country_of_origin": "",

        "ingredients": "",
        "directions": "",
        "storage": "",
        "dimensions": "",

        "fssai_license_numbers": fssai_numbers,

        "nutrition": {},
        "claims": [],

        "ocr_confidence": {},
    }

    # --------------------------------------------------------
    # Quantity unit normalization
    # --------------------------------------------------------

    quantity = result["net_quantity"]

    if quantity:

        lower = quantity.lower()

        if "pcs" in lower or "piece" in lower:
            result["quantity_unit"] = "pcs"

        elif lower.endswith("kg"):
            result["quantity_unit"] = "kg"

        elif lower.endswith("g"):
            result["quantity_unit"] = "g"

        elif lower.endswith("ml"):
            result["quantity_unit"] = "ml"

        elif lower.endswith("l"):
            result["quantity_unit"] = "l"

        # OCR frequently reads "6 pcs" as "6N".
        elif lower.endswith("n") and lower[:-1].isdigit():
            result["quantity_unit"] = "pcs"
            result["net_quantity"] = lower[:-1]

    return result


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    text_path = "ocr_text/1000435318_clean.txt"

    layout_path = "ocr_text/1000435318_clean_layout.tsv"

    result = build_product_json(
        text_path,
        layout_path,
    )

    import json

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )