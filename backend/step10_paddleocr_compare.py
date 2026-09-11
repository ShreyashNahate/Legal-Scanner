"""
COMPARISON TOOL: Tesseract (current pipeline) vs PaddleOCR-VL.

Why this exists:
Rather than guessing which OCR engine is "better," this runs BOTH on
the exact same image and prints both outputs side by side so you can
visually judge which one reads a specific real photo more accurately.

NOTE: PaddleOCR-VL downloads model weights (a few GB) from HuggingFace/
ModelScope/BOS the first time it runs. This requires your machine to
have normal internet access to those hosts - it could NOT be tested in
the development sandbox this project was built in, since that sandbox's
network is locked to a small allowlist that doesn't include those model
hosts. You are the first one actually running this.

Setup:
  pip install paddleocr "paddlex[ocr]"
  (first run will download model weights - may take a few minutes)

Usage:
  python3 step10_paddleocr_compare.py input_product/your_real_photo.jpg
"""

import sys
import time

from step6_ocr_product_image import ocr_product_image


def run_tesseract(image_path: str):
    print("=" * 70)
    print("TESSERACT (current pipeline - adaptive raw/preprocessed)")
    print("=" * 70)
    start = time.time()
    text_path, layout_path = ocr_product_image(image_path, "ocr_text")
    elapsed = time.time() - start

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\nTime taken: {elapsed:.1f}s")
    print(f"\nExtracted text:\n{text}")
    return text


def run_paddleocr_vl(image_path: str):
    print("\n" + "=" * 70)
    print("PADDLEOCR-VL")
    print("=" * 70)

    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        print("paddleocr is not installed. Run: pip install paddleocr \"paddlex[ocr]\"")
        return None

    try:
        start = time.time()
        pipeline = PaddleOCRVL(pipeline_version="v1")
        output = pipeline.predict(image_path)

        extracted_text_parts = []
        for res in output:
            res.save_to_json(save_path="paddleocr_output")
            res.save_to_markdown(save_path="paddleocr_output")
            # Pull readable text out of the result for direct comparison
            # with Tesseract's plain-text output.
            if hasattr(res, "markdown") and res.markdown:
                extracted_text_parts.append(str(res.markdown))

        elapsed = time.time() - start
        full_text = "\n".join(extracted_text_parts)

        print(f"\nTime taken: {elapsed:.1f}s")
        print(f"\nExtracted text (also saved to paddleocr_output/):\n{full_text}")
        return full_text

    except Exception as e:
        print(f"\nPaddleOCR-VL failed to run: {e}")
        print("This is most likely a model-download issue (network access to "
              "HuggingFace/ModelScope/BOS) or a missing dependency - check the "
              "error message above for specifics.")
        return None


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "input_product/sample_product_label.png"

    tesseract_text = run_tesseract(image_path)
    paddle_text = run_paddleocr_vl(image_path)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tesseract extracted {len(tesseract_text.strip())} characters.")
    if paddle_text is not None:
        print(f"PaddleOCR-VL extracted {len(paddle_text.strip())} characters.")
        print("\nCompare the two text blocks above manually - character count "
              "alone doesn't tell you which is more ACCURATE, just which read "
              "more text. Check specific fields (manufacturer name, net "
              "quantity, ingredients) in both outputs against the real label.")
    else:
        print("PaddleOCR-VL did not run successfully - see error above.")