"""
HYBRID MERGE: combine OCR+regex extraction with AI vision extraction.

Why this exists:
Neither extraction method alone is reliable on all real-world photos:
  - OCR+regex (step6+step7): free, fast, fully deterministic, but
    breaks when Tesseract misreads text or a label uses unexpected
    phrasing regex doesn't anticipate.
  - AI vision (step8): reads the image directly, handles messy/angled
    photos and unusual phrasing far better, but costs an API call and
    can occasionally miss a field or hit output-length limits on very
    dense labels.

This module merges the two: for every field, prefer the AI vision
result IF it found something; fall back to the OCR+regex result
otherwise. This is strictly additive - a field is only ever filled in
by whichever source actually found a value, never invented from
nothing. If BOTH sources come back empty for a field, it stays empty.

This does NOT change the compliance decision logic. rule_engine.py
still runs on whatever merged product.json comes out of this - it has
no idea whether a given field came from OCR or from AI.
"""


def _pick(vision_value, regex_value):
    """Prefer the vision value if it's non-empty; otherwise use regex value.
    Never invents a value neither source provided."""
    v = str(vision_value).strip() if vision_value is not None else ""
    r = str(regex_value).strip() if regex_value is not None else ""
    return v if v else r


def merge_product_data(product_vision: dict, product_regex: dict):
    """
    Merges two product dicts field-by-field, preferring vision's value
    when present. Also merges the nested 'nutrition' dict the same way,
    and unions the two 'claims' lists (deduplicated, case-insensitive).
    """
    merged = {}
    source_used = {}  # tracks which source actually supplied each field - useful for debugging/UI

    all_top_level_keys = set(product_vision.keys()) | set(product_regex.keys())
    all_top_level_keys.discard("nutrition")
    all_top_level_keys.discard("claims")
    all_top_level_keys.discard("ocr_source")

    for key in all_top_level_keys:
        vision_val = product_vision.get(key, "")
        regex_val = product_regex.get(key, "")
        merged[key] = _pick(vision_val, regex_val)
        source_used[key] = "vision" if str(vision_val).strip() else ("regex" if str(regex_val).strip() else "none")

    # ---- Nutrition sub-fields, same field-by-field preference ----
    vision_nutrition = product_vision.get("nutrition", {}) or {}
    regex_nutrition = product_regex.get("nutrition", {}) or {}
    all_nutrition_keys = set(vision_nutrition.keys()) | set(regex_nutrition.keys())

    merged_nutrition = {}
    for key in all_nutrition_keys:
        merged_nutrition[key] = _pick(vision_nutrition.get(key, ""), regex_nutrition.get(key, ""))
    merged["nutrition"] = merged_nutrition

    # ---- Claims: union of both sources, deduplicated ----
    vision_claims = product_vision.get("claims", []) or []
    regex_claims = product_regex.get("claims", []) or []

    seen_lower = set()
    merged_claims = []
    for claim in list(vision_claims) + list(regex_claims):
        claim_str = str(claim).strip()
        key = claim_str.lower()
        if claim_str and key not in seen_lower:
            seen_lower.add(key)
            merged_claims.append(claim_str)
    merged["claims"] = merged_claims

    merged["ocr_source"] = (
        f"hybrid(vision={product_vision.get('ocr_source', '?')}, "
        f"regex={product_regex.get('ocr_source', '?')})"
    )
    merged["_field_sources"] = source_used  # for debugging/transparency, not used by rule_engine

    return merged