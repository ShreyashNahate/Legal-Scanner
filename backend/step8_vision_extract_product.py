"""
STEP (ML/AI upgrade): Vision-model product label extraction using Groq.

Why Groq: it offers a genuinely free developer tier (no credit card,
gated only by rate limits) with vision-capable models, which makes it
a good fit for an SIH prototype with no budget.

Why this exists at all:
Regex-based extraction (step7_build_product_json.py) only works when
OCR text roughly matches expected phrasing. A vision-capable model can
look at the actual IMAGE and understand label layout the way a human
would, which handles messy real-world photos far better than fixed
regex patterns.

CRITICAL ARCHITECTURAL RULE - UNCHANGED FROM THE REST OF THE PROJECT:
This script ONLY extracts field VALUES from the image. It does NOT,
and must never, decide PASS/FAIL/compliant/non-compliant. That
decision stays 100% inside rule_engine.py's deterministic code. This
script just produces a different (hopefully more accurate) product.json
for rule_engine.py to consume - it changes nothing about HOW compliance
is decided.

The model is explicitly instructed to:
  - Only transcribe what is actually visible/legible on the label
  - Leave a field empty if it's not confidently readable
  - NEVER guess, infer, or invent a value
  - NEVER make any legal/compliance judgment

Setup:
  1. Get a free API key at https://console.groq.com (no credit card needed)
  2. Set it as an environment variable:  export GROQ_API_KEY="your-key-here"
  3. pip install groq

Input : a product label image path
Output: dataset/product_vision.json  (same schema as step7's product.json)
"""

import os
import sys
import json
import base64
import mimetypes

from groq import Groq

# qwen/qwen3.6-27b: 27B multimodal model, supports JSON mode, free tier.
# (qwen/qwen3.8-27b is a newer alternative with tunable reasoning effort,
# but limited to 3 images/request vs 5 - either works for this use case.)
MODEL_NAME = "qwen/qwen3.8-27b"

LEGAL_FIELDS_PROMPT = """Look at the product label image.

Extract the following fields exactly as printed on the label.
Do not guess or infer. If a field is not visible or readable, use an empty string.

Return ONLY valid JSON. Do not use markdown or explanations.

The JSON must contain exactly these keys:

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
  "expiry_date": "",
  "best_before": "",
  "batch_number": "",
  "consumer_care_email": "",
  "consumer_care_phone": "",
  "ingredients": ""
}

For ingredients, copy only text that is actually visible on the label."""

NUTRITION_CLAIMS_PROMPT = """You are looking at a photo of a packaged product's label - specifically its nutrition facts table and any marketing claims printed on it.

Extract ONLY the following, exactly as printed. Do not infer, guess, or invent any value. If a value is not visible/legible/present, leave it as an empty string "".

You are NOT deciding whether any claim is true or false. You are only transcribing what is literally printed.

Be concise: return COMPACT JSON (no indentation/whitespace), no markdown fences, no explanation - JSON only. List at most 5 claims.

Respond in JSON format matching exactly this shape:

{"nutrition":{"serving_size":"","calories":"","protein_g":"","carbohydrates_g":"","sugar_g":"","added_sugar_g":"","fat_g":"","sodium_mg":""},"claims":[]}"""


def _encode_image(image_path: str):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(
            f"Unsupported image type '{mime_type}' for '{image_path}'. "
            f"Use a .jpg, .png, or .webp file."
        )
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8"), mime_type


def _call_vision_model(client, image_data: str, mime_type: str, prompt: str):
    completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    },
                },
            ],
        }
    ],
    temperature=0.2,
    max_completion_tokens=1000,
    reasoning_effort="none",
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


def extract_product_fields_with_vision(image_path: str, api_key: str = None):
    """
    Makes TWO smaller, focused calls instead of one large one - this
    keeps each response comfortably within Groq's free-tier output
    token cap, since a single combined call (legal fields + full
    nutrition table + claims) was overflowing on dense real labels
    (ingredients lists competing with the nutrition table for the same
    1000-token budget).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find '{image_path}'.")

    client = Groq(api_key=api_key)  # falls back to GROQ_API_KEY env var
    image_data, mime_type = _encode_image(image_path)

    legal_fields = _call_vision_model(client, image_data, mime_type, LEGAL_FIELDS_PROMPT)
    nutrition_claims = _call_vision_model(client, image_data, mime_type, NUTRITION_CLAIMS_PROMPT)

    product = {**legal_fields, **nutrition_claims}
    product["ocr_source"] = f"vision_model:{MODEL_NAME}:{image_path}"
    return product


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "input_product/sample_product_label.png"
    output_path = "dataset/product_vision.json"

    print(f"Sending '{image_path}' to {MODEL_NAME} (Groq) for field extraction...")
    product = extract_product_fields_with_vision(image_path)

    print("\nExtracted fields:")
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
    print("\nThis product JSON can be fed into rule_engine.py exactly like "
          "the regex-based one:")
    print(f"  python3 rule_engine.py {output_path}")
