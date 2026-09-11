import os

# Thread limits BEFORE Paddle import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json

import paddle

try:
    paddle.set_flags({
        "FLAGS_use_mkldnn": False
    })
except Exception:
    pass

from paddleocr import PaddleOCR


print("Loading PaddleOCR...", file=sys.stderr, flush=True)

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("PaddleOCR READY", file=sys.stderr, flush=True)


def process_image(image_path):

    result = ocr.predict(image_path)

    all_text = []

    for page in result:

        data = page.json

        if isinstance(data, str):
            data = json.loads(data)

        res = data.get("res", {})

        texts = res.get("rec_texts", [])

        for text in texts:
            if text and text.strip():
                all_text.append(text.strip())

    return {
        "success": True,
        "text": all_text,
        "full_text": "\n".join(all_text)
    }


# ---------------------------------------------------------
# WORKER MODE
# ---------------------------------------------------------

print("OCR worker waiting...", file=sys.stderr, flush=True)

for line in sys.stdin:

    line = line.strip()

    if not line:
        continue

    try:

        request = json.loads(line)

        image_path = request["image_path"]

        print(
            f"Processing: {image_path}",
            file=sys.stderr,
            flush=True
        )

        output = process_image(image_path)

        print(
            json.dumps(output),
            flush=True
        )

        print(
            "OCR completed.",
            file=sys.stderr,
            flush=True
        )

    except Exception as e:

        print(
            json.dumps({
                "success": False,
                "error": str(e)
            }),
            flush=True
        )