"""
NUTRITION & CLAIMS ANALYSIS MODULE (kept separate from Legal Metrology).

Why this is a separate module (per project rules):
Legal Metrology compliance (rule_engine.py) checks mandatory
DECLARATIONS (manufacturer, quantity, MRP, etc.) against the Legal
Metrology Rules. This module is a different concern entirely:
  1. Extracting NUTRITION FACTS from the label (calories, protein, etc.)
  2. Detecting marketing CLAIMS ("supports muscle", "boosts focus", etc.)

These two concerns must never be mixed into the same PASS/FAIL logic
as Legal Metrology, and claims must NEVER be automatically labeled
illegal or false. We only ever say:
  SUPPORTED        - the claim has some matching evidence on the label itself
  NEEDS_VERIFICATION - the claim was detected but we have no way to
                        verify it from OCR text alone
(There is no automatic "NOT_VERIFIED = illegal" status — a human
always makes that call; this module is deterministic and neutral.)

Input : OCR text from a product label (same text step6/step7 use)
Output: {
    "nutrition": {...},
    "claims": [ {claim_text, category, status, evidence}, ... ]
}
"""

import re

# -----------------------------------------------------------------
# NUTRITION FIELD PATTERNS
# -----------------------------------------------------------------
# Same "search per line, don't invent missing values" approach as
# step7_build_product_json.py. Add more alternate phrasings as you
# test against real nutrition tables (e.g. "Energy" vs "Calories").
NUTRITION_PATTERNS = {
    "serving_size": r"Serving\s*Size\s*[:\-]?\s*(.+)",
    "calories": r"(?:Calories|Energy)\s*[:\-]?\s*([\d.]+)",
    "protein_g": r"Protein\s*[:\-]?\s*([\d.]+)\s*g",
    "carbohydrates_g": r"(?:Total\s*)?Carbohydrates?\s*[:\-]?\s*([\d.]+)\s*g",
    "sugar_g": r"(?:Total\s*)?Sugars?\s*[:\-]?\s*([\d.]+)\s*g",
    "fat_g": r"(?:Total\s*)?Fat\s*[:\-]?\s*([\d.]+)\s*g",
    "sodium_mg": r"Sodium\s*[:\-]?\s*([\d.]+)\s*mg",
}

# Some fields need MULTIPLE alternate patterns because real labels phrase
# them differently (e.g. the number can come before OR after the label
# text). Tried in order; first match wins.
ADDED_SUGAR_PATTERNS = [
    r"Added\s*Sugars?\s*[:\-]?\s*([\d.]+)\s*g",       # "Added Sugars: 2g"
    r"Includes\s+([\d.]+)\s*g\s+Added\s+Sugars?",      # "Includes 0g Added Sugars"
]

# Lines that signal we've moved past a multi-line block (ingredients,
# manufacturer info, etc.) and should stop appending further lines.
BLOCK_STOP_PATTERN = re.compile(
    r"^(CONTAINS|ALLERGEN|MANUFACTURED|PROCESSED\s+ON|MFG|EXP|BEST\s*BEFORE|"
    r"BATCH|NET\s*QUANTITY|MRP|NUTRITION\s*FACTS|SERVING\s*SIZE|CALORIES)",
    re.IGNORECASE,
)

# -----------------------------------------------------------------
# CLAIM DETECTION
# -----------------------------------------------------------------
# Each entry: (regex pattern, claim category, human-readable claim text)
# check_fn (optional): given the extracted nutrition dict, returns
# (status, evidence) if it can make a simple, defensible determination.
# If check_fn is None, or check_fn can't decide, the claim always falls
# back to NEEDS_VERIFICATION — we never guess.

def _check_protein_claim(nutrition: dict):
    protein = nutrition.get("protein_g")
    if protein is None or protein == "":
        return "NEEDS_VERIFICATION", "Claim detected but protein content not found on label to cross-check"
    try:
        protein_value = float(protein)
    except ValueError:
        return "NEEDS_VERIFICATION", f"Claim detected but protein value '{protein}' could not be parsed"

    # Simple, transparent heuristic threshold (not a legal standard -
    # just a sanity check a human reviewer could easily verify/adjust).
    if protein_value >= 10:
        return "SUPPORTED", f"Label declares {protein_value}g protein per serving, consistent with a protein-related claim"
    else:
        return "NEEDS_VERIFICATION", f"Label declares only {protein_value}g protein per serving; may not clearly support this claim"


CLAIM_PATTERNS = [
    (r"supports?\s+muscle", "muscle_growth", "Supports muscle", _check_protein_claim),
    (r"builds?\s+strength", "muscle_growth", "Builds strength", _check_protein_claim),
    (r"growth\s*(?:&|and)\s*recovery", "muscle_growth", "Growth & recovery", _check_protein_claim),
    (r"boosts?\s+focus", "cognitive", "Boosts focus", None),
    (r"mental\s+clarity", "cognitive", "Mental clarity", None),
    (r"rapid\s+absorption", "absorption", "Rapid absorption", None),
    (r"tested\s+for\s+heavy\s+metals", "safety_testing", "Tested for heavy metals", None),
    (r"immunity\s+booster", "immunity", "Immunity booster", None),
    (r"100\s*%\s*natural", "natural_claim", "100% natural", None),
    (r"no\s+added\s+sugar", "sugar_claim", "No added sugar", None),
]


def extract_nutrition_facts(text: str):
    """Same per-line regex approach as the Legal Metrology field extractor.
    Missing values stay empty — never invented."""
    lines = text.splitlines()
    nutrition = {}

    for field, pattern in NUTRITION_PATTERNS.items():
        found_value = None
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                found_value = match.group(1).strip()
                break
        nutrition[field] = found_value if found_value else ""

    # Added sugar needs multiple alternate patterns tried in order,
    # since real labels phrase it differently (number before OR after
    # the label text).
    added_sugar_value = None
    for pattern in ADDED_SUGAR_PATTERNS:
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                added_sugar_value = match.group(1).strip()
                break
        if added_sugar_value:
            break
    nutrition["added_sugar_g"] = added_sugar_value if added_sugar_value else ""

    # Ingredients often wrap across MULTIPLE OCR lines (one long comma
    # list on a real label). Capture the anchor line's remainder, then
    # keep appending following lines until a known stop-word line or a
    # blank line is hit — same technique used for manufacturer address
    # blocks in step7_build_product_json.py.
    nutrition["ingredients"] = _extract_ingredients_block(lines)

    return nutrition


def _extract_ingredients_block(lines: list):
    ingredients_parts = []
    found_anchor = False

    for i, line in enumerate(lines):
        anchor_match = re.match(r"\s*Ingredients\s*[:\-]?\s*(.*)", line, re.IGNORECASE)
        if anchor_match:
            found_anchor = True
            first_part = anchor_match.group(1).strip()
            if first_part:
                ingredients_parts.append(first_part)

            for next_line in lines[i + 1:]:
                stripped = next_line.strip()
                if stripped == "" or BLOCK_STOP_PATTERN.match(stripped):
                    break
                ingredients_parts.append(stripped)
            break

    if not found_anchor:
        return ""
    return " ".join(ingredients_parts).strip()


def detect_claims(text: str, nutrition: dict):
    """
    Scans the full text for known claim phrases. Each match gets a
    status of SUPPORTED or NEEDS_VERIFICATION - never an automatic
    accusation of a false/illegal claim.
    """
    claims_found = []

    for pattern, category, claim_text, check_fn in CLAIM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            if check_fn is not None:
                status, evidence = check_fn(nutrition)
            else:
                status = "NEEDS_VERIFICATION"
                evidence = "Claim detected on label; cannot be verified from OCR text alone - requires manual/lab verification"

            claims_found.append({
                "claim_text": claim_text,
                "category": category,
                "status": status,
                "evidence": evidence,
            })

    return claims_found


def analyze_nutrition_and_claims(text: str):
    """Main entry point for this module. Returns the combined result."""
    nutrition = extract_nutrition_facts(text)
    claims = detect_claims(text, nutrition)

    not_found = [k for k, v in nutrition.items() if v == ""]

    return {
        "nutrition": nutrition,
        "claims": claims,
        "nutrition_fields_not_found": not_found,
    }


if __name__ == "__main__":
    # Quick standalone test with a sample label text (not tied to any
    # specific product image - just for verifying the module works).
    sample_text = """
    SAMPLE PROTEIN BAR
    Serving Size: 60 g
    Calories: 220 kcal
    Protein: 15 g
    Total Carbohydrates: 20 g
    Sugars: 5 g
    Added Sugars: 2 g
    Total Fat: 8 g
    Sodium: 150 mg

    Supports muscle growth and recovery.
    Boosts focus and mental clarity.
    Tested for heavy metals.
    100% Natural. No added sugar claims on front panel.
    """

    result = analyze_nutrition_and_claims(sample_text)

    print("Nutrition facts extracted:")
    for k, v in result["nutrition"].items():
        marker = "  " if v else "  [EMPTY] "
        print(f"{marker}{k}: {v!r}")

    if result["nutrition_fields_not_found"]:
        print(f"\nNOTE: these nutrition fields were not found (left empty, "
              f"not invented): {result['nutrition_fields_not_found']}")

    print(f"\nClaims detected ({len(result['claims'])}):")
    for c in result["claims"]:
        print(f"  [{c['status']:<18}] \"{c['claim_text']}\" -> {c['evidence']}")