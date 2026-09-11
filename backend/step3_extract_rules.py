"""
STEP 3: Combine all page OCR text and extract individual Rules.

Why we do this:
Step 2 gave us raw OCR text per page. Now we need to find where each
"Rule" actually starts and ends, and pull out its number, title, and
body text — while also remembering which page(s) it came from.

IMPORTANT CAVEAT (see PROJECT PLAN in the master prompt):
The PDF may contain schedules/tables with numbers that LOOK like rule
numbers but are not. This script only treats a line as the START of a
new rule if it matches the pattern:

    Rule <number>. <title text...>

at the START of a line (after trimming whitespace). This is a simple,
deterministic heuristic — not perfect, but reliable enough for a
prototype. Anything that doesn't match this exact pattern is treated as
part of the body text of whichever rule came before it.

We do NOT invent or guess rule text. If a rule number is skipped
because OCR didn't produce a clean match, we report it clearly instead
of silently failing.

Input : ocr_text/page_1.txt, page_2.txt, ...
Output: dataset/raw_rules.json   (structured: number, title, text, source_pages)
        dataset/rules.txt        (human-readable combined rules)
"""

import os
import re
import json

# ---- CONFIG ----
INPUT_DIR = "ocr_text"
OUTPUT_JSON = "dataset/raw_rules.json"
OUTPUT_TXT = "dataset/rules.txt"

# We only try to extract rules in this range for the prototype.
# (Per the project plan: Rules 1-34 is sufficient; missing ones are
# reported rather than blocking the whole pipeline.)
MIN_RULE = 1
MAX_RULE = 34

# Matches lines like:
#   "Rule 6. Declarations to be made on every package."   (period)
#   "Rule 6 — Declarations to be made on every package"   (em-dash, real PDF)
#   "Rule 6 - Declarations..."                             (hyphen, en-dash)
# - "Rule" (case-insensitive)
# - a number
# - a separator: period, hyphen, en-dash (–), or em-dash (—)
# - then the rest of the title on the same line
RULE_HEADER_PATTERN = re.compile(
    r"^\s*Rule\s+(\d{1,3})\s*[.\-–—]\s*(.+)$", re.IGNORECASE
)


def natural_sort_key(filename: str):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else 0


def load_pages(input_dir: str):
    """Returns a list of (page_number, text) tuples, sorted by page number.

    IMPORTANT: only picks up files matching 'page_<number>.txt' — NOT
    every .txt file in the folder. This directory can also contain OCR
    output from product label images (e.g. from step6), and those must
    never be mixed into rule extraction.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Could not find '{input_dir}'. Run step2_ocr_pages.py first."
        )

    page_file_pattern = re.compile(r"^page_(\d+)\.txt$", re.IGNORECASE)
    txt_files = [f for f in os.listdir(input_dir) if page_file_pattern.match(f)]
    if not txt_files:
        raise FileNotFoundError(
            f"No 'page_<number>.txt' files found in '{input_dir}'. "
            f"(Other .txt files may be present but are ignored on purpose "
            f"— e.g. product label OCR output.)"
        )

    txt_files.sort(key=natural_sort_key)

    pages = []
    for filename in txt_files:
        page_num = natural_sort_key(filename)
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            pages.append((page_num, f.read()))
    return pages


def extract_rules(pages):
    """
    Walks through every page line by line. Whenever a line matches the
    "Rule N. Title" pattern, it starts a new rule. All following lines
    (until the next matching header) are appended as that rule's body text.
    """
    rules = []          # list of dicts, in the order they were found
    current_rule = None
    seen_numbers = set()
    skipped_out_of_range = []
    duplicate_numbers = []

    for page_num, page_text in pages:
        for raw_line in page_text.splitlines():
            match = RULE_HEADER_PATTERN.match(raw_line)

            if match:
                rule_number = int(match.group(1))
                title = match.group(2).strip()

                # Sanity check: ignore numbers way outside the expected
                # range (likely a schedule/table number, not a real rule).
                if not (MIN_RULE <= rule_number <= MAX_RULE):
                    skipped_out_of_range.append((rule_number, page_num, raw_line.strip()))
                    # Treat this line as body text of the current rule
                    # instead of starting a new one.
                    if current_rule:
                        current_rule["text_lines"].append(raw_line.strip())
                    continue

                if rule_number in seen_numbers:
                    duplicate_numbers.append((rule_number, page_num))
                    # Still start a new entry but flag it, so nothing
                    # gets silently merged/lost.
                seen_numbers.add(rule_number)

                # Close off the previous rule
                if current_rule:
                    rules.append(current_rule)

                current_rule = {
                    "rule_number": rule_number,
                    "title": title,
                    "text_lines": [],
                    "source_pages": [page_num],
                }
            else:
                if current_rule is None:
                    continue  # text before the first detected rule; skip
                stripped = raw_line.strip()
                if stripped:
                    current_rule["text_lines"].append(stripped)
                if page_num not in current_rule["source_pages"]:
                    current_rule["source_pages"].append(page_num)

    if current_rule:
        rules.append(current_rule)

    # Convert text_lines -> single text string
    for rule in rules:
        rule["text"] = " ".join(rule.pop("text_lines"))

    return rules, skipped_out_of_range, duplicate_numbers


def report_missing_rules(rules):
    found_numbers = {r["rule_number"] for r in rules}
    expected = set(range(MIN_RULE, MAX_RULE + 1))
    missing = sorted(expected - found_numbers)
    return missing


def save_outputs(rules, json_path, txt_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    rules_sorted = sorted(rules, key=lambda r: r["rule_number"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rules_sorted, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        for rule in rules_sorted:
            f.write(f"Rule {rule['rule_number']}. {rule['title']}\n")
            f.write(f"(Source page(s): {rule['source_pages']})\n")
            f.write(f"{rule['text']}\n")
            f.write("\n" + ("-" * 60) + "\n\n")


if __name__ == "__main__":
    pages = load_pages(INPUT_DIR)
    print(f"Loaded OCR text from {len(pages)} page(s).\n")

    rules, skipped, duplicates = extract_rules(pages)

    print(f"Extracted {len(rules)} rule(s).")

    if duplicates:
        print(f"\nWARNING: duplicate rule numbers detected (check OCR quality): {duplicates}")

    if skipped:
        print(f"\nNOTE: {len(skipped)} line(s) looked like 'Rule N.' but N was "
              f"outside the expected range ({MIN_RULE}-{MAX_RULE}), so they were "
              f"treated as body text, not new rules. First few examples:")
        for num, page, line in skipped[:5]:
            print(f"  page {page}: 'Rule {num}.' -> {line[:80]}")

    missing = report_missing_rules(rules)
    if missing:
        print(f"\nNOTE: these rule numbers were NOT found and may need manual "
              f"review (could be missing from OCR, or genuinely not in this "
              f"PDF section): {missing}")

    save_outputs(rules, OUTPUT_JSON, OUTPUT_TXT)
    print(f"\nSaved:\n  {OUTPUT_JSON}\n  {OUTPUT_TXT}")