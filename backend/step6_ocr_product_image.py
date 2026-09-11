"""
STEP 6 v4: Controlled multi-pass OCR for product label images.

Strategy:
    - Original OCR = PRIMARY / trusted base
    - Upscaled OCR = recovery source
    - Targeted region OCR = recovery source
    - Additional OCR is only allowed to add words that are
      not already represented by the primary OCR.
    - Avoids the huge duplication seen in v2/v3.

Outputs:
    ocr_text/<name>.txt
    ocr_text/<name>_layout.tsv
"""

import os
import csv
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PRODUCT_IMAGE_PATH = "input_product/sample_product_label.png"
OUTPUT_DIR = "ocr_text"
OCR_LANG = "eng"

UPSCALE_FACTOR = 2

# Regions overlap slightly.
REGION_OVERLAP = 0.10

# Maximum distance for considering two detections the same.
DUPLICATE_DISTANCE = 12

MIN_BOX_SIZE = 2


# ---------------------------------------------------------
# IMAGE LOADING
# ---------------------------------------------------------

def load_source_image(image_path: str) -> Image.Image:

    try:
        image = Image.open(image_path)
        image.load()

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        return image

    except Exception as e:
        raise RuntimeError(
            f"Could not load image for OCR: {e}"
        )


# ---------------------------------------------------------
# IMAGE PREPARATION
# ---------------------------------------------------------

def create_ocr_images(source_image: Image.Image):

    rgb = np.array(source_image)

    # Original image.
    original = rgb

    # 2x upscale.
    upscaled = cv2.resize(
        rgb,
        None,
        fx=UPSCALE_FACTOR,
        fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )

    # Contrast-enhanced grayscale.
    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_RGB2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    return {
        "original": original,
        "upscaled": upscaled,
        "enhanced": enhanced,
    }


# ---------------------------------------------------------
# TESSERACT
# ---------------------------------------------------------

def safe_confidence(value):

    try:
        return float(value)

    except Exception:
        return -1.0


def average_confidence(layout_data):

    values = []

    for i in range(
        len(layout_data["text"])
    ):

        text = layout_data["text"][i].strip()

        if not text:
            continue

        confidence = safe_confidence(
            layout_data["conf"][i]
        )

        if confidence >= 0:
            values.append(confidence)

    if not values:
        return 0.0

    return sum(values) / len(values)


def run_tesseract(
    image,
    lang="eng",
    psm=6,
):

    return pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--oem 3 --psm {psm}",
        output_type=pytesseract.Output.DICT,
    )


# ---------------------------------------------------------
# REGIONS
# ---------------------------------------------------------

def generate_regions(width, height):

    half_height = int(
        height * 0.52
    )

    step = int(
        half_height
        * (1 - REGION_OVERLAP)
    )

    regions = []

    # Always keep the entire image as primary.
    regions.append(
        (
            "full",
            0,
            0,
            width,
            height,
        )
    )

    y = 0
    index = 0

    while y < height:

        y2 = min(
            height,
            y + half_height,
        )

        regions.append(
            (
                f"section_{index}",
                0,
                y,
                width,
                y2,
            )
        )

        if y2 >= height:
            break

        y += step
        index += 1

    return regions


# ---------------------------------------------------------
# EXTRACT WORDS
# ---------------------------------------------------------

def extract_words(
    layout_data,
    offset_x,
    offset_y,
    scale,
    source,
    region,
):

    words = []

    for i in range(
        len(layout_data["text"])
    ):

        text = layout_data["text"][i].strip()

        if not text:
            continue

        confidence = safe_confidence(
            layout_data["conf"][i]
        )

        left = int(
            layout_data["left"][i]
        )

        top = int(
            layout_data["top"][i]
        )

        width = int(
            layout_data["width"][i]
        )

        height = int(
            layout_data["height"][i]
        )

        if (
            width < MIN_BOX_SIZE
            or height < MIN_BOX_SIZE
        ):
            continue

        original_left = int(
            offset_x
            + left / scale
        )

        original_top = int(
            offset_y
            + top / scale
        )

        original_width = max(
            1,
            int(width / scale)
        )

        original_height = max(
            1,
            int(height / scale)
        )

        words.append(
            {
                "word": text,
                "left": original_left,
                "top": original_top,
                "width": original_width,
                "height": original_height,
                "confidence": confidence,
                "source": source,
                "region": region,
            }
        )

    return words


# ---------------------------------------------------------
# WORD NORMALIZATION
# ---------------------------------------------------------

def normalize_word(text):

    text = text.lower().strip()

    # Remove punctuation around a word.
    text = text.strip(
        ".,:;!?|[](){}<>\"'`"
    )

    return text


# ---------------------------------------------------------
# DUPLICATE CHECK
# ---------------------------------------------------------

def center(word):

    return (
        word["left"]
        + word["width"] / 2,

        word["top"]
        + word["height"] / 2,
    )


def boxes_overlap(a, b):

    ax1 = a["left"]
    ay1 = a["top"]

    ax2 = (
        a["left"]
        + a["width"]
    )

    ay2 = (
        a["top"]
        + a["height"]
    )

    bx1 = b["left"]
    by1 = b["top"]

    bx2 = (
        b["left"]
        + b["width"]
    )

    by2 = (
        b["top"]
        + b["height"]
    )

    return (
        min(ax2, bx2)
        > max(ax1, bx1)
        and
        min(ay2, by2)
        > max(ay1, by1)
    )


def same_word(a, b):

    wa = normalize_word(
        a["word"]
    )

    wb = normalize_word(
        b["word"]
    )

    if not wa or not wb:
        return False

    if wa != wb:
        return False

    ax, ay = center(a)
    bx, by = center(b)

    distance = (
        (ax - bx) ** 2
        +
        (ay - by) ** 2
    ) ** 0.5

    if distance > DUPLICATE_DISTANCE:
        return False

    return boxes_overlap(
        a,
        b,
    )


# ---------------------------------------------------------
# ADD PRIMARY WORDS
# ---------------------------------------------------------

def add_primary_words(
    result,
    words,
):

    for word in words:

        result.append(word)


# ---------------------------------------------------------
# ADD RECOVERY WORDS
# ---------------------------------------------------------

def add_recovery_words(
    result,
    recovery_words,
):

    added = 0

    for candidate in recovery_words:

        duplicate = False

        for existing in result:

            if same_word(
                candidate,
                existing,
            ):
                duplicate = True
                break

        if duplicate:
            continue

        # -------------------------------------------------
        # Important:
        #
        # We don't want low-confidence garbage from
        # recovery OCR to overwhelm the primary OCR.
        #
        # Only accept reasonably confident recovery words.
        # -------------------------------------------------

        if (
            candidate["confidence"] >= 45
        ):

            result.append(
                candidate
            )

            added += 1

    return added


# ---------------------------------------------------------
# TEXT RECONSTRUCTION
# ---------------------------------------------------------

def build_text(words):

    if not words:
        return ""

    # Sort top-to-bottom first.
    words = sorted(
        words,
        key=lambda x: (
            x["top"],
            x["left"],
        )
    )

    lines = []

    current = []
    current_y = None

    for word in words:

        center_y = (
            word["top"]
            + word["height"] / 2
        )

        if current_y is None:

            current = [word]
            current_y = center_y

            continue

        tolerance = max(
            8,
            word["height"] * 0.65,
        )

        if abs(
            center_y - current_y
        ) <= tolerance:

            current.append(word)

            current_y = (
                current_y
                * (len(current) - 1)
                + center_y
            ) / len(current)

        else:

            current.sort(
                key=lambda x: x["left"]
            )

            lines.append(
                current
            )

            current = [word]
            current_y = center_y

    if current:

        current.sort(
            key=lambda x: x["left"]
        )

        lines.append(
            current
        )

    output = []

    for line in lines:

        output.append(
            " ".join(
                word["word"]
                for word in line
            )
        )

    return "\n".join(
        output
    )


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def ocr_product_image(
    image_path: str,
    output_dir: str,
    lang: str = "eng",
):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Could not find "
            f"'{image_path}'."
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    source_image = load_source_image(
        image_path
    )

    original_width, original_height = (
        source_image.size
    )

    print(
        f"\nOCR image size: "
        f"{original_width} x "
        f"{original_height}"
    )

    # -----------------------------------------------------
    # CREATE OCR IMAGES
    # -----------------------------------------------------

    images = create_ocr_images(
        source_image
    )

    print(
        "OCR images: "
        + ", ".join(
            images.keys()
        )
    )

    # -----------------------------------------------------
    # PRIMARY OCR
    # -----------------------------------------------------

    print(
        "\n========== PRIMARY OCR =========="
    )

    primary_layout = run_tesseract(
        images["original"],
        lang=lang,
        psm=6,
    )

    primary_confidence = (
        average_confidence(
            primary_layout
        )
    )

    primary_words = extract_words(
        primary_layout,
        0,
        0,
        1,
        "primary",
        "full",
    )

    print(
        f"Primary confidence: "
        f"{primary_confidence:.1f}/100"
    )

    print(
        f"Primary words: "
        f"{len(primary_words)}"
    )

    # -----------------------------------------------------
    # FINAL WORD LIST STARTS WITH PRIMARY OCR.
    # -----------------------------------------------------

    final_words = []

    add_primary_words(
        final_words,
        primary_words,
    )

    # -----------------------------------------------------
    # RECOVERY OCR #1
    # -----------------------------------------------------

    print(
        "\n========== UPSCALE RECOVERY =========="
    )

    upscaled_layout = run_tesseract(
        images["upscaled"],
        lang=lang,
        psm=6,
    )

    upscaled_confidence = (
        average_confidence(
            upscaled_layout
        )
    )

    upscaled_words = extract_words(
        upscaled_layout,
        0,
        0,
        UPSCALE_FACTOR,
        "upscaled",
        "full",
    )

    added = add_recovery_words(
        final_words,
        upscaled_words,
    )

    print(
        f"Upscaled confidence: "
        f"{upscaled_confidence:.1f}/100"
    )

    print(
        f"Upscaled detections: "
        f"{len(upscaled_words)}"
    )

    print(
        f"New words recovered: "
        f"{added}"
    )

    # -----------------------------------------------------
    # RECOVERY OCR #2
    #
    # Enhanced image is only used on sections.
    # -----------------------------------------------------

    print(
        "\n========== REGION RECOVERY =========="
    )

    regions = generate_regions(
        original_width,
        original_height,
    )

    enhanced = images["enhanced"]

    total_region_words = 0
    total_region_added = 0

    for (
        region_name,
        x1,
        y1,
        x2,
        y2,
    ) in regions:

        # Skip the "full" region because the original
        # and upscaled full-image passes already covered it.
        if region_name == "full":
            continue

        ux1 = int(
            x1 * UPSCALE_FACTOR
        )

        uy1 = int(
            y1 * UPSCALE_FACTOR
        )

        ux2 = int(
            x2 * UPSCALE_FACTOR
        )

        uy2 = int(
            y2 * UPSCALE_FACTOR
        )

        crop = enhanced[
            uy1:uy2,
            ux1:ux2
        ]

        if crop.size == 0:
            continue

        layout = run_tesseract(
            crop,
            lang=lang,
            psm=6,
        )

        confidence = (
            average_confidence(
                layout
            )
        )

        region_words = extract_words(
            layout,
            x1,
            y1,
            UPSCALE_FACTOR,
            "enhanced",
            region_name,
        )

        added = add_recovery_words(
            final_words,
            region_words,
        )

        total_region_words += (
            len(region_words)
        )

        total_region_added += added

        print(
            f"{region_name}: "
            f"{len(region_words)} detections, "
            f"{confidence:.1f} confidence, "
            f"{added} new"
        )

    # -----------------------------------------------------
    # FINAL DEDUPLICATION
    #
    # This is mostly a safety net because recovery words
    # were already checked before being added.
    # -----------------------------------------------------

    print(
        "\n========== FINAL MERGE =========="
    )

    print(
        f"Primary words: "
        f"{len(primary_words)}"
    )

    print(
        f"New from upscale: "
        f"{added if 'added' in locals() else 0}"
    )

    print(
        f"Region detections: "
        f"{total_region_words}"
    )

    print(
        f"Region words recovered: "
        f"{total_region_added}"
    )

    # -----------------------------------------------------
    # Build final text.
    # -----------------------------------------------------

    text = build_text(
        final_words
    )

    text_path = os.path.join(
        output_dir,
        f"{base_name}.txt"
    )

    with open(
        text_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(text)

    # -----------------------------------------------------
    # SAVE LAYOUT
    # -----------------------------------------------------

    layout_path = os.path.join(
        output_dir,
        f"{base_name}_layout.tsv"
    )

    with open(
        layout_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow([
            "word",
            "left",
            "top",
            "width",
            "height",
            "confidence",
        ])

        for word in sorted(
            final_words,
            key=lambda x: (
                x["top"],
                x["left"],
            )
        ):

            writer.writerow([
                word["word"],
                word["left"],
                word["top"],
                word["width"],
                word["height"],
                f"{word['confidence']:.1f}",
            ])

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    confidences = [
        word["confidence"]
        for word in final_words
        if word["confidence"] >= 0
    ]

    avg_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    low_confidence = sum(
        1
        for word in final_words
        if (
            0 <= word["confidence"] < 50
        )
    )

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    print(
        f"\nSaved plain text -> "
        f"{text_path} "
        f"({len(text.strip())} characters)"
    )

    print(
        f"Saved layout data -> "
        f"{layout_path} "
        f"({len(final_words)} words)"
    )

    print(
        f"\nFinal average confidence: "
        f"{avg_confidence:.1f}/100"
    )

    print(
        f"Low-confidence words (<50): "
        f"{low_confidence}"
    )

    print(
        "\n========== OCR PREVIEW =========="
    )

    print(
        text[:5000]
    )

    print(
        "=================================\n"
    )

    return (
        text_path,
        layout_path,
    )