"""
STEP 1: Convert every page of the Legal Metrology PDF into a PNG image.

Why we do this:
The PDF might contain scanned pages (basically photos of paper), not
selectable text. So instead of trying to extract text directly from the
PDF, we first turn every page into a plain image. Later steps will run
OCR (text recognition) on each image.

Input : input_pdf/<your_pdf_file>.pdf
Output: pdf_pages/page_1.png, page_2.png, ...
"""

import os
from pdf2image import convert_from_path

# ---- CONFIG (change this if your PDF has a different name) ----
INPUT_PDF = "input_pdf/Legal_Metrology_Packaged_Commodities_English_Rules_1-34.pdf"
OUTPUT_DIR = "pdf_pages"
DPI = 300  # higher DPI = clearer image = better OCR later, but slower


def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Could not find '{pdf_path}'. "
            f"Put your Legal Metrology PDF inside the input_pdf/ folder "
            f"and update INPUT_PDF in this script."
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading PDF: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=dpi)
    print(f"Detected {len(pages)} page(s). Converting to images at {dpi} DPI...")

    saved_paths = []
    for i, page in enumerate(pages, start=1):
        out_path = os.path.join(output_dir, f"page_{i}.png")
        page.save(out_path, "PNG")
        saved_paths.append(out_path)
        print(f"  Saved {out_path}")

    print(f"\nDone. {len(saved_paths)} page image(s) saved in '{output_dir}/'")
    return saved_paths


if __name__ == "__main__":
    convert_pdf_to_images(INPUT_PDF, OUTPUT_DIR, DPI)
