"""
STEP 6 (master plan Steps 12-15): OCR a product label image.

Why we do this:
Unlike the PDF (one clean page per image), a product label photo is
messier — different fonts, sizes, orientations, and multiple "sections"
(front label, back label, nutrition table, etc.) in one image.

We extract TWO things from OCR here, not just one:
  1. Plain text (like Step 2 did for PDF pages) -> ocr_text/product1.txt
  2. Word-level bounding boxes + confidence scores
     -> ocr_text/product1_layout.tsv

Why bounding boxes matter (per project plan):
Later, to check things like "is this on the Principal Display Panel"
or "is the font readable", we need to know WHERE text is on the image,
not just WHAT the text says. We are not building that logic yet — this
step only captures the raw coordinate + confidence data so it's
available when we need it.

We do NOT try to get perfect OCR here. Low-confidence words are kept
in the layout file (with their confidence score) rather than silently
dropped, so later steps can decide whether to trust them or mark
something as REVIEW.

Input : product image (jpg/png) — path set below
Output: ocr_text/<name>.txt            (plain text)
        ocr_text/<name>_layout.tsv     (word, bounding box, confidence)
"""

import os
import csv
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

# ---- CONFIG ----
PRODUCT_IMAGE_PATH = "input_product/sample_product_label.png"
OUTPUT_DIR = "ocr_text"
OCR_LANG = "eng"  # use "eng+hin" if the label has Hindi text too


def preprocess_for_ocr(image_path: str):
    """
    Improves OCR reliability on real-world photos (uneven lighting,
    shadows, varied backgrounds, slight blur) by converting to
    grayscale and applying adaptive thresholding, which is much more
    robust than relying on the raw color photo alone.

    Returns a PIL Image ready for pytesseract. Falls back to the
    original image (rather than crashing) if OpenCV can't read the
    file for any reason — we never want preprocessing to be a hard
    dependency that blocks OCR entirely.
    """
    cv_image = cv2.imread(image_path)
    if cv_image is None:
        # OpenCV couldn't decode it - fall back to letting PIL/pytesseract
        # try the raw file directly instead of failing outright.
        return Image.open(image_path)

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Denoise slightly before thresholding - helps with photo grain/noise
    # from phone cameras in low light.
    gray = cv2.medianBlur(gray, 3)

    # Adaptive thresholding handles uneven lighting/backgrounds far
    # better than a single global threshold would (e.g. a label
    # photographed with a shadow across part of it).
    thresholded = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=35, C=11,
    )

    return Image.fromarray(thresholded)


def _average_confidence(layout_data: dict) -> float:
    """Computes average word confidence from pytesseract's image_to_data output."""
    confidences = [
        int(layout_data["conf"][i])
        for i in range(len(layout_data["conf"]))
        if layout_data["text"][i].strip() != "" and int(layout_data["conf"][i]) >= 0
    ]
    return sum(confidences) / len(confidences) if confidences else 0.0


def ocr_product_image(image_path: str, output_dir: str, lang: str = "eng"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Could not find '{image_path}'. Put a product label image "
            f"there, or update PRODUCT_IMAGE_PATH in this script."
        )

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ---------------------------------------------------------
    # Load and fully decode the image with Pillow.
    # ---------------------------------------------------------
    try:
        source_image = Image.open(image_path)
        source_image.load()

        # Respect phone-camera EXIF orientation.
        source_image = ImageOps.exif_transpose(source_image)

        # Convert to RGB.
        source_image = source_image.convert("RGB")

    except Exception as e:
        raise RuntimeError(
            f"Could not load image for OCR: {e}"
        )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Save a temporary PNG instead of passing a PIL image
    # directly to pytesseract.
    #
    # This avoids pytesseract creating a temporary JPEG
    # which was causing Leptonica JPEG errors.
    # ---------------------------------------------------------
    temp_png_path = os.path.join(
        output_dir,
        f".{base_name}_ocr_input.png"
    )

    source_image.save(
        temp_png_path,
        format="PNG"
    )

    # Create preprocessed version.
    preprocessed_image = preprocess_for_ocr(image_path)

    temp_preprocessed_png_path = os.path.join(
        output_dir,
        f".{base_name}_ocr_preprocessed.png"
    )

    preprocessed_image.save(
        temp_preprocessed_png_path,
        format="PNG"
    )

    # ---------------------------------------------------------
    # Run OCR directly on PNG FILE PATHS.
    # Do not pass PIL images to pytesseract.
    # ---------------------------------------------------------
    candidates = {
        "raw": temp_png_path,
        "preprocessed": temp_preprocessed_png_path,
    }

    scores = {}
    layout_by_candidate = {}

    for name, candidate_path in candidates.items():

        layout_data = pytesseract.image_to_data(
            candidate_path,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )

        scores[name] = _average_confidence(layout_data)
        layout_by_candidate[name] = layout_data

    best_name = max(scores, key=scores.get)
    best_image_path = candidates[best_name]
    best_layout_data = layout_by_candidate[best_name]

    print(
        f"OCR confidence — raw: {scores['raw']:.1f}/100, "
        f"preprocessed: {scores['preprocessed']:.1f}/100 "
        f"-> using '{best_name}'"
    )

    # ---------------------------------------------------------
    # 1. Plain text extraction
    # ---------------------------------------------------------
    text = pytesseract.image_to_string(
        best_image_path,
        lang=lang,
    )

    text_path = os.path.join(
        output_dir,
        f"{base_name}.txt"
    )

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(
        f"  Saved plain text -> {text_path} "
        f"({len(text.strip())} characters)"
    )

    # ---------------------------------------------------------
    # 2. Word-level bounding boxes + confidence
    # ---------------------------------------------------------
    layout_path = os.path.join(
        output_dir,
        f"{base_name}_layout.tsv"
    )

    num_words_written = 0

    with open(
        layout_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "word",
            "left",
            "top",
            "width",
            "height",
            "confidence"
        ])

        num_boxes = len(best_layout_data["text"])

        for i in range(num_boxes):

            word = best_layout_data["text"][i].strip()

            if word == "":
                continue

            writer.writerow([
                word,
                best_layout_data["left"][i],
                best_layout_data["top"][i],
                best_layout_data["width"][i],
                best_layout_data["height"][i],
                best_layout_data["conf"][i],
            ])

            num_words_written += 1

    print(
        f"  Saved layout data -> {layout_path} "
        f"({num_words_written} word(s))"
    )

    avg_conf = scores[best_name]

    low_conf_count = sum(
        1
        for i in range(len(best_layout_data["conf"]))
        if (
            best_layout_data["text"][i].strip() != ""
            and 0 <= int(best_layout_data["conf"][i]) < 50
        )
    )

    print(
        f"\n  Average word confidence: "
        f"{avg_conf:.1f}/100"
    )

    if low_conf_count:
        print(
            f"  NOTE: {low_conf_count} word(s) had confidence "
            f"below 50. These may need manual review later "
            f"rather than being trusted blindly."
        )

    # ---------------------------------------------------------
    # Remove temporary PNG files.
    # ---------------------------------------------------------
    try:
        os.remove(temp_png_path)
        os.remove(temp_preprocessed_png_path)
    except OSError:
        pass

    return text_path, layout_path