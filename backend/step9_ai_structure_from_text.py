"""
SMOOTH FLOW MODULE: AI-powered structuring of already-OCR'd text.

Why this is a better flow than the pure-vision endpoint:
The vision endpoint (step8) asks one AI call to do TWO jobs at once:
  1. Read the pixels of the image (hard, slow, needs the whole image)
  2. Structure what it read into clean JSON fields (easy for a text model)

This module splits that into two separate, more reliable steps:
  1. OCR (step6) - already-tuned, adaptive raw/preprocessed pipeline -
     turns the image into plain text
  2. THIS MODULE - a fast text-only AI model reads that OCR text and
     structures it into the same JSON schema, correcting obvious OCR
     noise ("Ruie" -> "Rule", stray punctuation) with much more
     consistency than fixed regex patterns, since it understands
     context/meaning rather than matching fixed string shapes.

This is the RECOMMENDED end-to-end flow:
  Image -> OCR (step6) -> AI text structuring (this file) -> rule_engine.py

CRITICAL ARCHITECTURAL RULE - UNCHANGED:
This module ONLY restructures/cleans already-extracted text into JSON
fields. It does NOT decide compliance. It is explicitly instructed to
leave a field empty rather than invent a value, and to never make any
legal/compliance judgment. rule_engine.py's deterministic code is the
only thing that ever produces a PASS/FAIL/REVIEW status.

Setup: same as step8 - a free GROQ_API_KEY from console.groq.com.
Uses a text-only model (faster and cheaper than the vision model,
since it only has to read text, not decode an image).
"""

import os
import json

from groq import Groq

# Fast general-purpose text model - no image decoding needed here, so
# this can be lighter/faster than the vision model used in step8.
MODEL_NAME = "qwen/qwen3.8-27b"

LEGAL_FIELDS_TEMPLATE = """You are given raw OCR text extracted from a photo of a packaged product's label. OCR is imperfect - it may have misread some characters, dropped words, or broken lines oddly.

Your job: read this OCR text and structure the LEGAL/PACKAGING fields below. You may correct OBVIOUS OCR noise (e.g. "Manufaeturer" is clearly "Manufacturer", "l0g" is clearly "10g") when confident, but NEVER invent information not present in the text. If a field is not present, leave it as an empty string "".

You are NOT deciding whether this product is compliant with any regulation. You are only structuring what the OCR text already contains.

Be concise: return COMPACT JSON (no indentation/whitespace), no markdown fences, no explanation - JSON only. For "ingredients", if there are more than 15 items, include only the first 15 followed by "...".

--- OCR TEXT START ---
{ocr_text}
--- OCR TEXT END ---

Respond in JSON format matching exactly this shape:

{{"product_name":"","common_or_generic_name":"","manufacturer_name":"","manufacturer_address":"","packer_name":"","importer_name":"","country_of_origin":"","net_quantity":"","quantity_unit":"","mrp":"","manufacturing_date":"","expiry_date":"","best_before":"","batch_number":"","consumer_care_email":"","consumer_care_phone":"","ingredients":""}}"""

NUTRITION_CLAIMS_TEMPLATE = """You are given raw OCR text extracted from a photo of a packaged product's label. OCR is imperfect - it may have misread some characters or broken lines oddly.

Your job: find and structure ONLY the NUTRITION FACTS and any MARKETING CLAIMS in this text. You may correct OBVIOUS OCR noise when confident, but NEVER invent a value not present in the text. If a value is not present, leave it as an empty string "".

You are NOT deciding whether any claim is true or false. You are only structuring what the text already contains.

Be concise: return COMPACT JSON (no indentation/whitespace), no markdown fences, no explanation - JSON only. List at most 5 claims.

--- OCR TEXT START ---
{ocr_text}
--- OCR TEXT END ---

Respond in JSON format matching exactly this shape:

{{"nutrition":{{"serving_size":"","calories":"","protein_g":"","carbohydrates_g":"","sugar_g":"","added_sugar_g":"","fat_g":"","sodium_mg":""}},"claims":[]}}"""


def _call_text_model(client, prompt: str):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # very low - faithful structuring, not creative writing
        max_completion_tokens=1000,  # Groq free tier hard cap
        response_format={"type": "json_object"},
    )

    raw_text = completion.choices[0].message.content.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON despite JSON mode being "
            f"enabled. Raw output was:\n{raw_text}\n\nError: {e}"
        )


def structure_text_with_ai(ocr_text: str, api_key: str = None):
    """
    Makes TWO smaller, focused calls instead of one large one - same
    fix as step8's vision extraction. A single combined call (legal
    fields + full nutrition table + claims) was overflowing Groq's
    free-tier output token cap on dense real labels.
    """
    if not ocr_text or not ocr_text.strip():
        raise ValueError("Empty OCR text provided - nothing for the AI to structure.")

    client = Groq(api_key=api_key)  # falls back to GROQ_API_KEY env var

    legal_fields = _call_text_model(client, LEGAL_FIELDS_TEMPLATE.format(ocr_text=ocr_text))
    nutrition_claims = _call_text_model(client, NUTRITION_CLAIMS_TEMPLATE.format(ocr_text=ocr_text))

    product = {**legal_fields, **nutrition_claims}
    product["ocr_source"] = f"ai_text_structuring:{MODEL_NAME}"
    return product


if __name__ == "__main__":
    import sys

    ocr_text_path = sys.argv[1] if len(sys.argv) > 1 else "ocr_text/sample_product_label.txt"
    output_path = "dataset/product_ai_structured.json"

    with open(ocr_text_path, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    print(f"Sending OCR text from '{ocr_text_path}' to {MODEL_NAME} (Groq) for structuring...")
    product = structure_text_with_ai(ocr_text)

    print("\nStructured fields:")
    for key, value in product.items():
        if key in ("nutrition", "claims", "ocr_source"):
            continue
        marker = "  " if str(value).strip() else "  [EMPTY] "
        print(f"{marker}{key}: {value!r}")

    print("\nNutrition:")
    for key, value in product.get("nutrition", {}).items():
        marker = "  " if str(value).strip() else "  [EMPTY] "
        print(f"{marker}{key}: {value!r}")

    print(f"\nClaims found: {product.get('claims', [])}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(product, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")