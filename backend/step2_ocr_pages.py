"""
STEP 2: Run OCR on every page image and save the extracted text.

Why we do this:
Now that each PDF page is a plain image (from Step 1), we run Tesseract
OCR on each one to pull out the actual text. We save one .txt file per
page so we can inspect exactly what was read from each page before we
try to combine everything and find "Rules" in Step 3.

Input : pdf_pages/page_1.png, page_2.png, ...
Output: ocr_text/page_1.txt, page_2.txt, ...
"""

import os
import re
import pytesseract
from PIL import Image

# ---- CONFIG ----
INPUT_DIR = "pdf_pages"
OUTPUT_DIR = "ocr_text"
# 'eng' for English only. Use 'eng+hin' if the PDF has Hindi pages too.
OCR_LANG = "eng"


def natural_sort_key(filename: str):
    """Sorts page_2.png before page_10.png (not alphabetically)."""
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else 0


def ocr_all_pages(input_dir: str, output_dir: str, lang: str = "eng"):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Could not find '{input_dir}'. Run step1_pdf_to_images.py first."
        )

    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    if not image_files:
        raise FileNotFoundError(f"No .png images found in '{input_dir}'.")

    image_files.sort(key=natural_sort_key)
    print(f"Found {len(image_files)} page image(s). Running OCR (lang='{lang}')...\n")

    results = []
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        page_num = natural_sort_key(filename)

        image = Image.open(img_path)
        text = pytesseract.image_to_string(image, lang=lang)

        out_filename = f"page_{page_num}.txt"
        out_path = os.path.join(output_dir, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        char_count = len(text.strip())
        print(f"  {filename} -> {out_filename}  ({char_count} characters extracted)")
        results.append((page_num, char_count))

    print(f"\nDone. OCR text saved in '{output_dir}/'")

    # Quick sanity warning: flag pages where OCR found almost nothing
    empty_pages = [p for p, c in results if c < 5]
    if empty_pages:
        print(
            f"\nWARNING: page(s) {empty_pages} produced almost no text. "
            f"These may be blank pages, or the image quality/DPI may need "
            f"to be increased in Step 1."
        )

    return results


if __name__ == "__main__":
    ocr_all_pages(INPUT_DIR, OUTPUT_DIR, OCR_LANG)
