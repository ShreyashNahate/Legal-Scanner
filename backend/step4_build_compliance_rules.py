"""
STEP 4: Build compliance_rules.json — the checkable subset of rules.

Why we do this:
dataset/raw_rules.json (from Step 3) has ALL extracted rules, in their
original descriptive legal wording. That's great for reference, but a
deterministic rule engine (Step 11) needs something more actionable:
"check whether field X is present/correctly formatted."

This script does NOT invent new legal requirements. It only:
  1. Looks at a curated list of rule numbers that map to a checkable
     product-label field (defined below, based on what's commonly
     checked in Legal Metrology compliance: manufacturer info, net
     quantity, MRP, consumer care, etc.)
  2. Pulls the REAL title/text for that rule number from
     dataset/raw_rules.json (the actual OCR-extracted source of truth)
  3. Attaches a "field" name (used later to look up the value in the
     product's structured JSON) and a "check_type" (used later by the
     rule engine to decide HOW to check it)

If a rule number in the curated list is missing from raw_rules.json
(e.g. OCR didn't extract it), this script reports that clearly instead
of inventing placeholder text.

Input : dataset/raw_rules.json
Output: dataset/compliance_rules.json
"""

import os
import json

INPUT_JSON = "dataset/raw_rules.json"
OUTPUT_JSON = "dataset/compliance_rules.json"

# -----------------------------------------------------------------
# CURATED CHECKLIST
# -----------------------------------------------------------------
# This list defines WHICH extracted rules become actionable compliance
# checks, and HOW the rule engine should check them later.
#
# check_type meanings (used by rule_engine.py in Step 11):
#   "presence"       -> field must simply be non-empty
#   "format"         -> field must be non-empty AND match an expected pattern
#   "review_only"    -> can't be reliably auto-checked from OCR text alone
#                        (e.g. needs layout/placement info) -> always REVIEW
#
# field: the key name we expect in the product JSON (from Step 16/product.json)
#
# You can add/remove rule numbers here as the project develops — this
# is meant to be a living, editable checklist, not something the
# script re-derives automatically (to avoid the engine silently
# checking things nobody reviewed).
CURATED_CHECKLIST = [
    {"rule_number": 6,  "field": "mandatory_declarations", "check_type": "review_only"},
    {"rule_number": 7,  "field": "principal_display_panel", "check_type": "review_only"},
    {"rule_number": 8,  "field": "declaration_placement",   "check_type": "review_only"},
    {"rule_number": 9,  "field": "declaration_manner",      "check_type": "review_only"},
    {"rule_number": 10, "field": "manufacturer_name",       "check_type": "presence"},
    {"rule_number": 10, "field": "manufacturer_address",    "check_type": "presence"},
    {"rule_number": 11, "field": "net_quantity",            "check_type": "presence"},
    {"rule_number": 12, "field": "quantity_format",         "check_type": "format"},
    {"rule_number": 13, "field": "quantity_unit",           "check_type": "format"},
]


def load_raw_rules(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run step3_extract_rules.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    # index by rule_number for quick lookup
    return {r["rule_number"]: r for r in rules}


def build_compliance_rules(raw_rules_by_number: dict, checklist: list):
    compliance_rules = []
    missing = []

    for item in checklist:
        rule_number = item["rule_number"]
        source_rule = raw_rules_by_number.get(rule_number)

        if source_rule is None:
            missing.append(rule_number)
            continue  # do NOT invent a rule that wasn't actually extracted

        compliance_rules.append({
            "rule_number": rule_number,
            "field": item["field"],
            "check_type": item["check_type"],
            "title": source_rule["title"],
            "source_text": source_rule["text"],
            "source_pages": source_rule["source_pages"],
        })

    return compliance_rules, missing


def save_output(compliance_rules: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(compliance_rules, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raw_rules_by_number = load_raw_rules(INPUT_JSON)
    print(f"Loaded {len(raw_rules_by_number)} extracted rule(s) from {INPUT_JSON}\n")

    compliance_rules, missing = build_compliance_rules(raw_rules_by_number, CURATED_CHECKLIST)

    print(f"Built {len(compliance_rules)} compliance check(s):")
    for c in compliance_rules:
        print(f"  Rule {c['rule_number']} -> field='{c['field']}' "
              f"check_type='{c['check_type']}'")

    if missing:
        print(f"\nWARNING: these curated rule numbers were NOT found in "
              f"raw_rules.json (so no compliance check was created for "
              f"them): {sorted(set(missing))}")
        print("This usually means Step 3 didn't extract that rule. "
              "Check dataset/rules.txt or re-run Step 3 first.")

    save_output(compliance_rules, OUTPUT_JSON)
    print(f"\nSaved: {OUTPUT_JSON}")
