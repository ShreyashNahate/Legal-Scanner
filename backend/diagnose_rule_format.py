"""
DIAGNOSTIC SCRIPT (not part of the main pipeline)

Purpose: Step 3 found 0 rules, which means the real OCR text doesn't
match our assumed "Rule 6. Title..." pattern exactly. Instead of
guessing why, this script scans every OCR page and prints every line
that contains the word "rule" (case-insensitive) along with which page
it came from, so we can see the ACTUAL format the OCR produced.

Run this and paste the output back — it'll tell us exactly how to fix
the regex in step3_extract_rules.py.
"""

import os
import re

INPUT_DIR = "ocr_text"


def natural_sort_key(filename: str):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else 0


def main():
    txt_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt")]
    txt_files.sort(key=natural_sort_key)

    total_matches = 0
    for filename in txt_files:
        page_num = natural_sort_key(filename)
        path = os.path.join(INPUT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if re.search(r"rule", line, re.IGNORECASE):
                total_matches += 1
                # repr() shows hidden whitespace/characters clearly
                print(f"[page {page_num}] {repr(line.strip())}")

    print(f"\nTotal lines containing 'rule' (case-insensitive): {total_matches}")

    if total_matches == 0:
        print(
            "\nNo lines contain the word 'rule' at all. This likely means:\n"
            "  - OCR quality is poor (check pdf_pages/*.png visually), or\n"
            "  - the DPI in Step 1 was too low, or\n"
            "  - the PDF's actual rule-numbering word is different "
            "(e.g. it might just be numbered '6.' without the word 'Rule')."
        )


if __name__ == "__main__":
    main()