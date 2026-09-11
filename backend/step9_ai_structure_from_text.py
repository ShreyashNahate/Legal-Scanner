"""
STEP 9: AI structuring of OCR text.

Flow:

    Product Image
          ↓
       Step 6 OCR
          ↓
    OCR text + layout
          ↓
     Qwen 3.8 27B
          ↓
    Structured JSON
          ↓
     rule_engine.py

IMPORTANT:
This module ONLY extracts/structures information.

It does NOT:
    - decide compliance
    - produce PASS/FAIL/REVIEW
    - invent missing values
    - apply legal rules

The deterministic rule engine remains responsible for compliance.
"""

import os
import json
from typing import Optional

from groq import Groq


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_NAME = "qwen/qwen3.8-27b"


# ---------------------------------------------------------
# LEGAL / PACKAGING EXTRACTION
# ---------------------------------------------------------

LEGAL_FIELDS_TEMPLATE = """
You are a HIGH-ACCURACY information extraction system for packaged-product labels.

You are given:
1. OCR text extracted from a real product label.
2. OCR layout information containing detected words and their positions.

Your ONLY job is to extract product/package/legal-label information from the supplied OCR.

============================================================
STRICT RULES
============================================================

1. NEVER invent information.

2. NEVER use outside knowledge.

3. NEVER make a compliance decision.

4. NEVER return PASS, FAIL, or REVIEW.

5. NEVER guess a number.

6. Preserve digits exactly as they appear in the OCR unless an
   OCR error is extremely obvious and the surrounding label text
   confirms the correction.

7. If a value is genuinely not present, return "".

8. If a label exists but its value is difficult to read, use the
   value that is actually present in OCR. Do not create a
   plausible replacement.

9. Search the ENTIRE OCR text before deciding that a field is empty.

10. The OCR may be badly ordered. Do NOT assume that a value must
    appear immediately after its label.

11. Use OCR LAYOUT positions to connect nearby labels and values
    when the plain OCR order is confusing.

============================================================
LEGAL FIELD SEARCH STRATEGY
============================================================

For EVERY field below, actively search the OCR for the relevant
label and its variants.

------------------------------------------------------------
PRODUCT NAME
------------------------------------------------------------

Look for:
- product name
- prominent product title
- brand/product description

Do not confuse marketing claims with the product name.

------------------------------------------------------------
COMMON / GENERIC NAME
------------------------------------------------------------

Look for wording such as:
- common name
- generic name
- description of product

Example:
"Nutrient-enriched Eggs" may be the product name while "Eggs"
may be the generic/common name.

------------------------------------------------------------
MANUFACTURER
------------------------------------------------------------

Search for:
- Manufactured by
- Mfd. by
- Mfd By
- Manufactured By
- manufacturer

Extract the company name associated with that label.

Do NOT automatically treat:
- Brand Owned By
- Marketed By
- Packed By
- Distributed By

as the manufacturer.

------------------------------------------------------------
MANUFACTURER ADDRESS
------------------------------------------------------------

Search the ENTIRE OCR for an address associated with the
manufacturer.

Addresses may contain:
- house/building numbers
- office numbers
- road names
- floor numbers
- locality
- city
- state
- PIN code

Example OCR fragments such as:

"401/402, Global Square"
"4th Floor"
"Deccan College Road"
"Yerwada"
"Pune"
"Maharashtra"
"411006"

may belong together as one complete address even if OCR
places them on separate lines.

Return the COMPLETE address supported by the OCR.

Do NOT shorten it unnecessarily.

------------------------------------------------------------
PACKER
------------------------------------------------------------

Search for:
- Packed By
- Packed by
- Packer
- Packed & Marketed by

Extract the company associated with the packing statement.

------------------------------------------------------------
IMPORTER
------------------------------------------------------------

Search for:
- Imported by
- Importer
- Imported By

If no importer is present, return "".

------------------------------------------------------------
COUNTRY OF ORIGIN
------------------------------------------------------------

Search for:
- Country of Origin
- Made in
- Product of

Do not assume India merely because an Indian address appears.

------------------------------------------------------------
NET QUANTITY
------------------------------------------------------------

Search for:
- Net Quantity
- Net Qty
- Quantity
- Net Content

Extract the numeric quantity and unit.

Examples:
"6 pcs" → quantity = "6", unit = "pcs"
"500 g" → quantity = "500", unit = "g"
"1 L" → quantity = "1", unit = "L"

Do not confuse serving size with net quantity.

------------------------------------------------------------
MRP
------------------------------------------------------------

This field is extremely important.

Search the ENTIRE OCR for:
- MRP
- M.R.P.
- Maximum Retail Price
- Max Retail Price
- Retail Price
- ₹
- Rs.
- INR

The MRP value may appear:
- on the same line
- on a different line
- before or after the label
- close to batch/date information

Use OCR layout positions to find the value nearest to the
MRP label when possible.

Preserve the numeric value exactly.

Do NOT calculate MRP.

Do NOT infer MRP from another price.

If the OCR contains an MRP label but the numeric value is
unreadable, return "".

------------------------------------------------------------
MANUFACTURING / PACKING DATE
------------------------------------------------------------

Search for:
- Mfg.
- Mfd.
- Mfg Date
- Mfd Date
- Manufacturing Date
- Date of Manufacture
- Pkg.
- Pkg Date
- Packed Date
- Packing Date

IMPORTANT:

Do not confuse:
- Batch number
- Manufacturing date
- Packing date
- Expiry date
- Use By date

with each other.

If the label uses "Pkg. Date", treat it as the package/packing
date and use it for manufacturing_date ONLY if the label context
clearly indicates that it represents the production/packing date.

Preserve the original date exactly.

------------------------------------------------------------
EXPIRY / USE BY
------------------------------------------------------------

Search for:
- Use By
- Use by
- Expiry
- Expiry Date
- Exp. Date
- Best Before

Do NOT automatically treat "Best Before" as an exact expiry date.

If an exact "Use By" or "Expiry" date is present, put it in
expiry_date.

If only "Best Before" information is present, put it in
best_before.

------------------------------------------------------------
BATCH NUMBER
------------------------------------------------------------

Search for:
- Batch No.
- Batch Number
- Batch
- Lot No.
- Lot Number
- Lot

Do NOT confuse a batch number with a date.

For example, if OCR contains:

"Batch No. 23/05"

then return:

"batch_number": "23/05"

Do not convert it into a date.

------------------------------------------------------------
FSSAI
------------------------------------------------------------

Search the ENTIRE OCR for:
- FSSAI
- FSSAI Lic. No.
- FSSAI License
- Lic. No.

If one or more FSSAI license numbers are visible, preserve
their digits exactly.

The current JSON does not have a dedicated FSSAI field, so
DO NOT invent a new field outside the required JSON schema.

------------------------------------------------------------
CONSUMER CARE
------------------------------------------------------------

Search for:
- Customer Care
- Consumer Care
- Consumer Complaints
- Contact
- Helpline
- Feedback

Extract:
- phone number
- email address

Preserve phone digits exactly.

Preserve email spelling exactly when readable.

------------------------------------------------------------
INGREDIENTS
------------------------------------------------------------

Search for:
- Ingredients
- Ingredient
- Contains

Extract only ingredients explicitly present.

Do NOT infer ingredients from the type of product.

============================================================
NUMBER PRESERVATION
============================================================

Numbers must be copied carefully.

Examples of numbers that must NOT be silently changed:

FSSAI:
11524996000376

Phone:
9130098788

PIN:
411006

Batch:
23/05

Date:
03/08/26

If OCR says:
"11524996000376"

do NOT output:
"11524996000378"

If OCR is uncertain, leave the field empty rather than
inventing digits.

============================================================
OCR LAYOUT
============================================================

The OCR layout contains positional information.

When OCR text is out of order:

1. Find the relevant label.
2. Look for nearby detected values.
3. Use the spatial relationship between the label and value.
4. Prefer a nearby value over a distant unrelated value.
5. Use the plain OCR text as additional evidence.

This is especially important for:
- MRP
- Batch No.
- Pkg. Date
- Use By
- Net Quantity
- FSSAI numbers
- addresses

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use EXACTLY this JSON structure:

{
  "product_name": "",
  "common_or_generic_name": "",
  "manufacturer_name": "",
  "manufacturer_address": "",
  "packer_name": "",
  "importer_name": "",
  "country_of_origin": "",
  "net_quantity": "",
  "quantity_unit": "",
  "mrp": "",
  "manufacturing_date": "",
  "packing_date": "",
  "expiry_date": "",
  "best_before": "",
  "batch_number": "",
  "fssai_license_numbers": [],
  "consumer_care_email": "",
  "consumer_care_phone": "",
  "ingredients": ""
}

Do not add explanations.

Do not add markdown.

Do not add extra fields.

============================================================
OCR TEXT
============================================================

{ocr_text}

============================================================
OCR LAYOUT
============================================================

{ocr_layout}
"""


# ---------------------------------------------------------
# NUTRITION + CLAIMS
# ---------------------------------------------------------

NUTRITION_CLAIMS_TEMPLATE = """
You are an information extraction system for packaged-product labels.

You are given OCR text extracted from a real product label.

Your task is ONLY to extract:

1. Nutrition information
2. Marketing/product claims

DO NOT:
- invent nutrition values
- calculate missing values
- convert units unless the OCR explicitly provides them
- decide whether a claim is true
- decide compliance
- produce PASS/FAIL/REVIEW

OCR from tables may be badly ordered.

Use the surrounding row labels to associate values with the correct nutrient.

IMPORTANT:

If a nutrition value is unclear, do NOT guess.

For example:

OCR:
"Protein ... 11.78"

Return:
"protein_g": "11.78"

If OCR contains:
"<0.50"

Preserve:
"<0.50"

Do not turn it into:
"0.50"

If units are unclear, preserve the visible value rather than inventing a unit.

Focus on these nutrition fields:

- serving size
- calories / energy
- protein
- carbohydrates
- sugar
- added sugar
- fat
- sodium

If other nutrients are clearly present, they may be included in an additional "other_nutrients" object.

Claims should contain short text phrases actually present on the package.

Examples:
- "Nutrient-enriched Eggs"
- "In-house Feed"
- "Infertile Eggs"
- "Contributes to the normal function of the immune system"

Do not create claims that are not present.

Return ONLY JSON.

Required JSON shape:

{
  "nutrition": {
    "serving_size": "",
    "calories": "",
    "protein_g": "",
    "carbohydrates_g": "",
    "sugar_g": "",
    "added_sugar_g": "",
    "fat_g": "",
    "sodium_mg": "",
    "other_nutrients": {}
  },
  "claims": []
}

OCR TEXT:
{ocr_text}

OCR LAYOUT:
{layout_text}
"""


# ---------------------------------------------------------
# JSON SCHEMA
# ---------------------------------------------------------

LEGAL_SCHEMA = {
    "name": "legal_product_fields",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,

        "properties": {
            "product_name": {
                "type": "string"
            },

            "common_or_generic_name": {
                "type": "string"
            },

            "manufacturer_name": {
                "type": "string"
            },

            "manufacturer_address": {
                "type": "string"
            },

            "packer_name": {
                "type": "string"
            },

            "importer_name": {
                "type": "string"
            },

            "country_of_origin": {
                "type": "string"
            },

            "net_quantity": {
                "type": "string"
            },

            "quantity_unit": {
                "type": "string"
            },

            "mrp": {
                "type": "string"
            },

            "manufacturing_date": {
                "type": "string"
            },

            "packing_date": {
                "type": "string"
            },

            "expiry_date": {
                "type": "string"
            },

            "best_before": {
                "type": "string"
            },

            "batch_number": {
                "type": "string"
            },

            "fssai_license_numbers": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },

            "consumer_care_email": {
                "type": "string"
            },

            "consumer_care_phone": {
                "type": "string"
            },

            "ingredients": {
                "type": "string"
            }
        },

        "required": [
            "product_name",
            "common_or_generic_name",
            "manufacturer_name",
            "manufacturer_address",
            "packer_name",
            "importer_name",
            "country_of_origin",
            "net_quantity",
            "quantity_unit",
            "mrp",
            "manufacturing_date",
            "packing_date",
            "expiry_date",
            "best_before",
            "batch_number",
            "fssai_license_numbers",
            "consumer_care_email",
            "consumer_care_phone",
            "ingredients"
        ]
    }
}
NUTRITION_SCHEMA = {
    "name": "nutrition_and_claims",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {

            "nutrition": {
                "type": "object",
                "additionalProperties": False,
                "properties": {

                    "serving_size": {
                        "type": "string"
                    },

                    "calories": {
                        "type": "string"
                    },

                    "protein_g": {
                        "type": "string"
                    },

                    "carbohydrates_g": {
                        "type": "string"
                    },

                    "sugar_g": {
                        "type": "string"
                    },

                    "added_sugar_g": {
                        "type": "string"
                    },

                    "fat_g": {
                        "type": "string"
                    },

                    "sodium_mg": {
                        "type": "string"
                    },

                    "other_nutrients": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {

                                "name": {
                                    "type": "string"
                                },

                                "per_100g": {
                                    "type": "string"
                                },

                                "per_serving": {
                                    "type": "string"
                                },

                                "unit": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "name",
                                "per_100g",
                                "per_serving",
                                "unit"
                            ]
                        }
                    }
                },

                "required": [
                    "serving_size",
                    "calories",
                    "protein_g",
                    "carbohydrates_g",
                    "sugar_g",
                    "added_sugar_g",
                    "fat_g",
                    "sodium_mg",
                    "other_nutrients"
                ]
            },

            "claims": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },

        "required": [
            "nutrition",
            "claims"
        ]
    }
}

# ---------------------------------------------------------
# MODEL CALL
# ---------------------------------------------------------

def _call_text_model(
    client,
    prompt: str,
    schema: dict,
):
    """
    Call Qwen using strict structured output.
    """

    completion = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],

        temperature=0.0,

        max_completion_tokens=900,

        reasoning_effort="none",

        response_format={
            "type": "json_schema",
            "json_schema": schema,
        },
    )

    raw_text = (
        completion
        .choices[0]
        .message
        .content
        .strip()
    )

    try:
        return json.loads(
            raw_text
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            "Qwen returned invalid JSON.\n\n"
            f"Raw output:\n{raw_text}\n\n"
            f"Error: {e}"
        )


# ---------------------------------------------------------
# LOAD LAYOUT
# ---------------------------------------------------------

def _load_layout(
    layout_path: Optional[str]
) -> str:

    if not layout_path:
        return ""

    if not os.path.exists(
        layout_path
    ):
        return ""

    try:

        with open(
            layout_path,
            "r",
            encoding="utf-8",
        ) as f:

            return f.read()

    except Exception:
        return ""


# ---------------------------------------------------------
# MAIN STRUCTURING FUNCTION
# ---------------------------------------------------------

def structure_text_with_ai(
    ocr_text: str,
    api_key: str = None,
    layout_path: str = None,
):
    """
    Structure OCR text into product JSON.

    layout_path is optional so existing API code continues
    working without modification.

    If supplied, the TSV layout is also provided to the model.
    """

    if not ocr_text or not ocr_text.strip():
        raise ValueError(
            "Empty OCR text provided - "
            "nothing for the AI to structure."
        )

    # -----------------------------------------------------
    # Create Groq client
    # -----------------------------------------------------

    client = Groq(
        api_key=api_key or os.getenv("GROQ_API_KEY")
    )

    # -----------------------------------------------------
    # Load OCR layout
    # -----------------------------------------------------

    # -----------------------------------------------------
# Load OCR layout
# -----------------------------------------------------

    layout_text = _load_layout(layout_path)

    if not layout_text:
        layout_text = "No OCR layout file was supplied."
    else:
        layout_lines = layout_text.splitlines()

        # Keep the header.
        selected_lines = layout_lines[:1]

        # Keywords that are especially useful for legal fields.
        keywords = [
            "mrp",
            "batch",
            "pkg",
            "mfd",
            "mfg",
            "manufact",
            "expiry",
            "use",
            "best",
            "net",
            "quantity",
            "fssai",
            "packed",
            "packer",
            "address",
            "phone",
            "mob",
            "email",
            "mail",
            "pune",
            "maharashtra",
            "411006",
        ]

        for line in layout_lines[1:]:
            line_lower = line.lower()

            if any(
                keyword in line_lower
                for keyword in keywords
            ):
                selected_lines.append(line)

        # Limit the final layout size.
        layout_text = "\n".join(
            selected_lines[:250]
        )

    # -----------------------------------------------------
    # Call 1: Legal / package fields
    # -----------------------------------------------------

    print(
        f"[step9] Extracting legal/package "
        f"fields using {MODEL_NAME}..."
    )

    legal_prompt = (
        LEGAL_FIELDS_TEMPLATE
        .replace("{ocr_text}", ocr_text)
        .replace("{ocr_layout}", layout_text)
    )

    legal_fields = _call_text_model(
        client,
        legal_prompt,
        LEGAL_SCHEMA,
    )

    # -----------------------------------------------------
    # Call 2: Nutrition + claims
    # -----------------------------------------------------

    print(
        f"[step9] Extracting nutrition/claims "
        f"using {MODEL_NAME}..."
    )

    nutrition_prompt = (
        NUTRITION_CLAIMS_TEMPLATE
        .replace("{ocr_text}", ocr_text)
        .replace("{ocr_layout}", layout_text)
    )

    nutrition_claims = _call_text_model(
        client,
        nutrition_prompt,
        NUTRITION_SCHEMA,
    )

    # -----------------------------------------------------
    # Merge both AI responses
    # -----------------------------------------------------

    product = {
        **legal_fields,
        **nutrition_claims,
    }

    product["ocr_source"] = (
        f"ai_text_structuring:{MODEL_NAME}"
    )

    return product


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys

    ocr_text_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "ocr_text/sample_product_label.txt"
    )

    layout_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else None
    )

    output_path = (
        "dataset/product_ai_structured.json"
    )

    with open(
        ocr_text_path,
        "r",
        encoding="utf-8",
    ) as f:

        ocr_text = f.read()

    print(
        f"Sending OCR text from "
        f"'{ocr_text_path}' to "
        f"{MODEL_NAME}..."
    )

    if layout_path:
        print(
            f"Using OCR layout: "
            f"{layout_path}"
        )

    product = structure_text_with_ai(
        ocr_text,
        layout_path=layout_path,
    )

    # -----------------------------------------------------
    # Print legal fields.
    # -----------------------------------------------------

    print(
        "\n========== LEGAL FIELDS =========="
    )

    for key, value in product.items():

        if key in (
            "nutrition",
            "claims",
            "ocr_source",
        ):
            continue

        if str(value).strip():

            print(
                f"{key}: {value}"
            )

        else:

            print(
                f"{key}: [EMPTY]"
            )

    # -----------------------------------------------------
    # Nutrition.
    # -----------------------------------------------------

    print(
        "\n========== NUTRITION =========="
    )

    nutrition = product.get(
        "nutrition",
        {}
    )

    for key, value in nutrition.items():

        if key == "other_nutrients":

            continue

        if str(value).strip():

            print(
                f"{key}: {value}"
            )

        else:

            print(
                f"{key}: [EMPTY]"
            )

    # -----------------------------------------------------
    # Other nutrients.
    # -----------------------------------------------------


    print("\n========== OTHER NUTRIENTS ==========")

    other_nutrients = product.get(
        "nutrition",
        {}
    ).get(
        "other_nutrients",
        []
    )

    for nutrient in other_nutrients:
        print(
            f"{nutrient.get('name', '')}: "
            f"per_100g={nutrient.get('per_100g', '')}, "
            f"per_serving={nutrient.get('per_serving', '')}, "
            f"unit={nutrient.get('unit', '')}"
        ) 

    # -----------------------------------------------------
    # Claims.
    # -----------------------------------------------------

    print(
        "\n========== CLAIMS =========="
    )

    for claim in product.get(
        "claims",
        []
    ):

        print(
            f"- {claim}"
        )

    # -----------------------------------------------------
    # Save JSON.
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            product,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved: {output_path}"
    )