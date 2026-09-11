"""
STEP 5 (master plan calls this "Step 11"): Deterministic Rule Engine.

Why we do this:
This is the core compliance engine. It takes:
  - dataset/compliance_rules.json  (the checks, built in Step 4)
  - a product JSON                 (structured product label data)
and produces a PASS / FAIL / REVIEW result for each check.

IMPORTANT (per project rules):
This engine is 100% deterministic code — no LLM is used to decide
PASS/FAIL. An LLM could later be used to explain a FAIL in plain
language, but never to make the compliance decision itself.

We also NEVER accuse anyone of fraud. Wording stays neutral:
  "Declaration not detected", "Could not verify", etc.

For this step, we test the engine against a MANUALLY CREATED product
JSON (dataset/sample_product.json) — not a real OCR'd product image
yet. That comes later (Steps 12-17 in the master plan).

Input : dataset/compliance_rules.json
        dataset/sample_product.json  (or any product JSON you point it at)
Output: dataset/compliance_report.json
        printed summary in the terminal
"""

import os
import re
import json
import sys

COMPLIANCE_RULES_PATH = "dataset/compliance_rules.json"
# Default now points at the REAL product.json built automatically by
# step7_build_product_json.py. You can still override it by passing a
# different path as a command-line argument, e.g.:
#   python3 rule_engine.py dataset/sample_product.json
DEFAULT_PRODUCT_JSON_PATH = "dataset/product.json"
OUTPUT_REPORT_PATH = "dataset/compliance_report.json"

# Statuses used throughout the engine
PASS = "PASS"
FAIL = "FAIL"
REVIEW = "REVIEW"
NOT_APPLICABLE = "NOT_APPLICABLE"

# -----------------------------------------------------------------
# FORMAT PATTERNS
# -----------------------------------------------------------------
# Used only when check_type == "format". Add more fields/patterns here
# as the project grows. Keep these simple and explainable.
FORMAT_PATTERNS = {
    # e.g. "60 g", "1.5 kg", "250 ml" -> number + space + unit
    "quantity_format": re.compile(r"^\s*\d+(\.\d+)?\s?(g|kg|ml|l|mg|cm|mm|pieces|units)\s*$", re.IGNORECASE),
    # a standalone valid unit string
    "quantity_unit": re.compile(r"^(g|kg|ml|l|mg|cm|mm|pieces|units)$", re.IGNORECASE),
}


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_presence(value: str):
    """PASS if the field has real content, FAIL if empty/missing."""
    if value is None:
        return FAIL, "Field missing from product data"
    value_str = str(value).strip()
    if value_str == "" or value_str.upper() == "REVIEW":
        return FAIL, "Declaration not detected"
    return PASS, "Declaration detected"


def check_format(field: str, value: str):
    """
    PASS if present AND matches the expected pattern for this field.
    FAIL if present but badly formatted, or missing entirely.
    """
    if value is None or str(value).strip() == "":
        return FAIL, "Declaration not detected"

    pattern = FORMAT_PATTERNS.get(field)
    if pattern is None:
        # No pattern defined yet for this field -> can't verify format,
        # so hand it to a human instead of guessing.
        return REVIEW, f"No format pattern defined yet for field '{field}'; requires manual verification"

    if pattern.match(str(value).strip()):
        return PASS, "Declaration present and correctly formatted"
    else:
        return FAIL, f"Declaration present but format could not be verified: '{value}'"


def run_check(compliance_rule: dict, product: dict):
    field = compliance_rule["field"]
    check_type = compliance_rule["check_type"]
    detected_value = product.get(field, "")

    if check_type == "presence":
        status, evidence = check_presence(detected_value)
    elif check_type == "format":
        status, evidence = check_format(field, detected_value)
    elif check_type == "review_only":
        status = REVIEW
        evidence = "This check requires layout/placement information not available from text alone; requires manual verification"
    else:
        status = REVIEW
        evidence = f"Unknown check_type '{check_type}'; requires manual verification"

    return {
        "rule_number": compliance_rule["rule_number"],
        "title": compliance_rule["title"],
        "field": field,
        "check_type": check_type,
        "status": status,
        "detected_value": detected_value,
        "evidence": evidence,
    }


def run_rule_engine(compliance_rules: list, product: dict):
    results = [run_check(rule, product) for rule in compliance_rules]

    summary = {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["status"] == PASS),
        "failed": sum(1 for r in results if r["status"] == FAIL),
        "review": sum(1 for r in results if r["status"] == REVIEW),
        "not_applicable": sum(1 for r in results if r["status"] == NOT_APPLICABLE),
    }

    return {
        "product_name": product.get("product_name", ""),
        "summary": summary,
        "results": results,
    }


def print_report(report: dict):
    print(f"Product: {report['product_name']}\n")
    for r in report["results"]:
        print(f"  Rule {r['rule_number']:>2} [{r['status']:<6}] field='{r['field']}' "
              f"-> {r['evidence']}")

    s = report["summary"]
    print(f"\nSummary: {s['total_checks']} checks | "
          f"{s['passed']} PASS | {s['failed']} FAIL | "
          f"{s['review']} REVIEW | {s['not_applicable']} N/A")


if __name__ == "__main__":
    # Allow: python3 rule_engine.py [path/to/product.json]
    product_json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRODUCT_JSON_PATH

    compliance_rules = load_json(COMPLIANCE_RULES_PATH)
    product = load_json(product_json_path)

    print(f"Loaded {len(compliance_rules)} compliance check(s) and product "
          f"'{product.get('product_name') or '(no product_name in JSON)'}' "
          f"from '{product_json_path}'\n")

    report = run_rule_engine(compliance_rules, product)
    print_report(report)

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_REPORT_PATH}")