"""
STEP 11 — SMART HYBRID MERGE

Combines three extraction sources:

1. OCR + Regex  (Step 6 + Step 7)
   - Best for exact compliance-sensitive values.
   - Deterministic.
   - No AI hallucination.

2. AI Vision    (Step 8)
   - Best for understanding the image visually.
   - Useful for product name, manufacturer, address, etc.

3. OCR + AI     (Step 9)
   - AI structures the OCR text and OCR layout.
   - Useful when regex misses fields.
   - Can understand OCR text better than regex alone.

Important:
For compliance-critical values, OCR/regex gets priority.
AI is used as a fallback or for contextual information.

The rule engine receives ONLY the final merged product.
"""


def _is_present(value):
    """
    Check whether a value contains useful data.

    Handles:
      - None
      - empty strings
      - empty lists
      - empty dictionaries
    """

    if value is None:
        return False

    if isinstance(value, (list, dict)):
        return len(value) > 0

    return bool(str(value).strip())


def _clean(value):
    """Return a clean string representation."""

    if value is None:
        return ""

    return str(value).strip()


def _choose_exact(
    regex_value,
    ai_text_value,
    vision_value,
):
    """
    Exact/compliance fields.

    Priority:

        Regex
          ↓
        AI text
          ↓
        Vision

    Regex is preferred because it is deterministic.
    """

    if _is_present(regex_value):
        return regex_value, "regex"

    if _is_present(ai_text_value):
        return ai_text_value, "ai_text"

    if _is_present(vision_value):
        return vision_value, "vision"

    return "", "none"


def _choose_context(
    vision_value,
    ai_text_value,
    regex_value,
):
    """
    Contextual fields.

    Priority:

        Vision
          ↓
        AI text
          ↓
        Regex
    """

    if _is_present(vision_value):
        return vision_value, "vision"

    if _is_present(ai_text_value):
        return ai_text_value, "ai_text"

    if _is_present(regex_value):
        return regex_value, "regex"

    return "", "none"


def _choose_nutrition(
    vision_value,
    ai_text_value,
    regex_value,
):
    """
    Nutrition fields.

    Priority:

        Vision
          ↓
        AI text
          ↓
        Regex

    Vision gets priority because it sees the actual nutrition
    table and its visual column structure.

    AI text is second because Step 9 receives OCR + layout.

    Regex is the final fallback.
    """

    if _is_present(vision_value):
        return vision_value, "vision"

    if _is_present(ai_text_value):
        return ai_text_value, "ai_text"

    if _is_present(regex_value):
        return regex_value, "regex"

    return "", "none"


def _merge_claims(
    product_vision,
    product_ai_text,
    product_regex,
):
    """
    Combine claims from all three sources.

    Duplicate claims are removed case-insensitively.
    """

    vision_claims = (
        product_vision.get("claims", [])
        or []
    )

    ai_text_claims = (
        product_ai_text.get("claims", [])
        or []
    )

    regex_claims = (
        product_regex.get("claims", [])
        or []
    )

    merged_claims = []
    seen = set()

    # Vision first because it generally understands
    # complete visual phrases better.
    all_claims = (
        list(vision_claims)
        + list(ai_text_claims)
        + list(regex_claims)
    )

    for claim in all_claims:

        if isinstance(claim, dict):
            claim_text = (
                claim.get("claim_text")
                or claim.get("text")
                or ""
            )
        else:
            claim_text = claim

        claim_text = _clean(claim_text)

        if not claim_text:
            continue

        normalized = claim_text.lower()

        if normalized not in seen:
            seen.add(normalized)
            merged_claims.append(claim_text)

    return merged_claims


def merge_product_data(
    product_vision: dict,
    product_regex: dict,
    product_ai_text: dict | None = None,
):
    """
    Merge:

        Step 8 Vision
        Step 7 Regex
        Step 9 OCR + AI

    into one final product object.
    """

    product_vision = product_vision or {}
    product_regex = product_regex or {}
    product_ai_text = product_ai_text or {}

    merged = {}
    source_used = {}

    # =====================================================
    # EXACT / COMPLIANCE-SENSITIVE FIELDS
    # =====================================================

    exact_fields = {
        "mrp",
        "batch_number",
        "manufacturing_date",
        "packing_date",
        "expiry_date",
        "best_before",
        "net_quantity",
        "quantity_unit",
        "consumer_care_phone",
        "consumer_care_email",
        "fssai_license_numbers",
    }

    # =====================================================
    # CONTEXTUAL FIELDS
    # =====================================================

    contextual_fields = {
        "product_name",
        "common_or_generic_name",
        "manufacturer_name",
        "manufacturer_address",
        "packer_name",
        "importer_name",
        "country_of_origin",
        "ingredients",
    }

    # =====================================================
    # COLLECT ALL TOP-LEVEL KEYS
    # =====================================================

    all_keys = (
        set(product_vision.keys())
        | set(product_regex.keys())
        | set(product_ai_text.keys())
    )

    all_keys.discard("nutrition")
    all_keys.discard("claims")
    all_keys.discard("ocr_source")
    all_keys.discard("_field_sources")

    # =====================================================
    # MERGE TOP-LEVEL FIELDS
    # =====================================================

    for key in sorted(all_keys):

        vision_value = product_vision.get(
            key,
            ""
        )

        regex_value = product_regex.get(
            key,
            ""
        )

        ai_text_value = product_ai_text.get(
            key,
            ""
        )

        # -------------------------------------------------
        # EXACT FIELDS
        # -------------------------------------------------

        if key in exact_fields:

            value, source = _choose_exact(
                regex_value,
                ai_text_value,
                vision_value,
            )

        # -------------------------------------------------
        # CONTEXTUAL FIELDS
        # -------------------------------------------------

        elif key in contextual_fields:

            value, source = _choose_context(
                vision_value,
                ai_text_value,
                regex_value,
            )

        # -------------------------------------------------
        # UNKNOWN FIELDS
        # -------------------------------------------------

        else:

            value, source = _choose_context(
                vision_value,
                ai_text_value,
                regex_value,
            )

        merged[key] = value
        source_used[key] = source

    # =====================================================
    # NUTRITION
    # =====================================================

    vision_nutrition = (
        product_vision.get("nutrition", {})
        or {}
    )

    regex_nutrition = (
        product_regex.get("nutrition", {})
        or {}
    )

    ai_text_nutrition = (
        product_ai_text.get("nutrition", {})
        or {}
    )

    nutrition_keys = (
        set(vision_nutrition.keys())
        | set(regex_nutrition.keys())
        | set(ai_text_nutrition.keys())
    )

    merged_nutrition = {}

    for key in sorted(nutrition_keys):

        vision_value = vision_nutrition.get(
            key,
            ""
        )

        regex_value = regex_nutrition.get(
            key,
            ""
        )

        ai_text_value = ai_text_nutrition.get(
            key,
            ""
        )

        value, source = _choose_nutrition(
            vision_value,
            ai_text_value,
            regex_value,
        )

        merged_nutrition[key] = value

        source_used[
            f"nutrition.{key}"
        ] = source

    merged["nutrition"] = merged_nutrition

    # =====================================================
    # CLAIMS
    # =====================================================

    merged["claims"] = _merge_claims(
        product_vision,
        product_ai_text,
        product_regex,
    )

    source_used["claims"] = "merged"

    # =====================================================
    # SOURCE INFORMATION
    # =====================================================

    vision_source = product_vision.get(
        "ocr_source",
        "not_available"
    )

    regex_source = product_regex.get(
        "ocr_source",
        "not_available"
    )

    ai_text_source = product_ai_text.get(
        "ocr_source",
        "not_available"
    )

    merged["ocr_source"] = (
        "hybrid("
        f"vision={vision_source}, "
        f"regex={regex_source}, "
        f"ai_text={ai_text_source}"
        ")"
    )

    # Keep this for debugging.
    # It tells us which source supplied every field.
    merged["_field_sources"] = source_used

    # =====================================================
    # DEBUG SUMMARY
    # =====================================================

    print("\n[hybrid] Field source summary:")

    for field, source in sorted(
        source_used.items()
    ):
        print(
            f"  {field}: {source}"
        )

    return merged