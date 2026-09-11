"""
Legal Metrology Compliance Scanner - Backend API.

THE FULL PIPELINE, in one place, for the PRIMARY endpoint (/scan-product):

    Image
      |
      v
    OCR (step6 - adaptive raw/preprocessed Tesseract pipeline)
      |
      v
    Field extraction (step7 - regex-based, deterministic, no AI needed)
      |
      +----> Legal Metrology fields -> rule_engine.py -> PASS/FAIL/REVIEW
      |       (deterministic code - this is what actually DECIDES compliance)
      |
      +----> Nutrition & claims (nutrition_analysis.py - separate concern,
              never affects the PASS/FAIL decision)

This is the endpoint that actually validates against the rules - not
just OCR text extraction. If you only see extracted text with no PASS/
FAIL/REVIEW verdicts, you are not calling this endpoint.

Two additional endpoints exist for AI-assisted extraction (useful for
messy photos where regex struggles) - see /scan-product-ai and
/scan-product-ai-vision below. All three endpoints run the SAME
rule_engine.py validation step; only HOW fields get extracted differs.

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000

Test with:
  curl -X POST "http://127.0.0.1:8000/scan-product" \
       -F "file=@input_product/sample_product_label.png"
"""

import os
import io

from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from step6_ocr_product_image import ocr_product_image
from step7_build_product_json import build_product_json, load_text
from rule_engine import load_json, run_rule_engine, COMPLIANCE_RULES_PATH
from nutrition_analysis import analyze_nutrition_and_claims
from step11_hybrid_merge import merge_product_data
from step9_ai_structure_from_text import structure_text_with_ai

from step1_pdf_to_images import convert_pdf_to_images
from step2_ocr_pages import ocr_all_pages
from step3_extract_rules import load_pages, extract_rules, report_missing_rules, save_outputs as save_rule_outputs
from step4_build_compliance_rules import load_raw_rules, build_compliance_rules, save_output as save_compliance_rules, CURATED_CHECKLIST

app = FastAPI(title="Legal Metrology Compliance Scanner API")

# Prototype-only setting: allows requests from any origin (needed for
# Flutter Web / Chrome testing). For a real deployment, restrict this
# to known origins instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
from history_routes import router as history_router
app.include_router(history_router)
UPLOAD_DIR = "input_product/api_uploads"
OCR_OUTPUT_DIR = "ocr_text"

ADMIN_PDF_UPLOAD_DIR = "input_pdf/api_uploads"
PDF_PAGES_DIR = "pdf_pages"
RAW_RULES_JSON = "dataset/raw_rules.json"
RULES_TXT = "dataset/rules.txt"


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Legal Metrology Compliance Scanner API is running"}


def _build_nutrition_response(text_or_product, from_text: bool):
    """Shared helper so all three scan endpoints return nutrition_analysis
    in the exact same shape, regardless of extraction method."""
    if from_text:
        return analyze_nutrition_and_claims(text_or_product)
    else:
        product = text_or_product
        return {
            "nutrition": product.get("nutrition", {}),
            "claims": [
                {"claim_text": c, "category": "unclassified", "status": "NEEDS_VERIFICATION",
                 "evidence": "Detected by AI; not cross-checked against nutrition values"}
                for c in product.get("claims", [])
            ],
            "nutrition_fields_not_found": [
                k for k, v in product.get("nutrition", {}).items() if not v
            ],
        }


def _save_upload(filename: str, contents: bytes) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as f:
        f.write(contents)

    return image_path

async def _validate_and_save_image(file: UploadFile) -> str:
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400,
            detail="Only .png, .jpg, or .jpeg product label images are supported.",
        )

    contents = await file.read()

    if len(contents) < 1024:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The uploaded image appears incomplete or empty "
                f"({len(contents)} bytes). Please try uploading it again."
            ),
        )

    # ---------------------------------------------------------
    # Decode and normalize the uploaded image.
    # This protects OCR and Groq Vision from malformed JPEGs.
    # ---------------------------------------------------------
    try:
        image = Image.open(io.BytesIO(contents))

        # Force Pillow to fully decode the uploaded image.
        image.load()

        # Respect phone-camera EXIF rotation.
        image = ImageOps.exif_transpose(image)

        # Convert PNG/JPEG/etc. into standard RGB JPEG.
        image = image.convert("RGB")

        output_buffer = io.BytesIO()

        image.save(
            output_buffer,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        clean_contents = output_buffer.getvalue()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode the uploaded image: {e}",
        )

    # Save normalized image as a new JPEG.
    base_name = os.path.splitext(
        os.path.basename(file.filename)
    )[0]

    clean_filename = f"{base_name}_clean.jpg"

    return _save_upload(
        clean_filename,
        clean_contents,
    )

@app.post("/scan-product")
async def scan_product(file: UploadFile = File(...)):
    """
    PRIMARY ENDPOINT - the full pipeline, regex-based (fast, free, no
    external API key needed):

        Image -> OCR -> regex field extraction -> rule_engine.py PASS/FAIL/REVIEW
                                                 -> nutrition/claims analysis

    Response shape:
    {
      "product": { manufacturer_name, net_quantity, mrp, mfg/exp dates,
                   batch_number, ingredients, ... },
      "report": { summary: {passed, failed, review, total_checks},
                  results: [ {rule_number, title, status, evidence}, ... ] },
      "nutrition_analysis": { nutrition: {...}, claims: [...] }
    }
    """
    image_path = await _validate_and_save_image(file)

    try:
        print(f"[scan] Running OCR on '{image_path}'...")
        text_path, layout_path = ocr_product_image(image_path, OCR_OUTPUT_DIR)
        text = load_text(text_path)

        print(f"[scan] Extracting fields via regex...")
        product = build_product_json(text_path)

        print(f"[scan] Validating against Legal Metrology rules...")
        compliance_rules = load_json(COMPLIANCE_RULES_PATH)
        report = run_rule_engine(compliance_rules, product)
        print(f"[scan] Result: {report['summary']}")

        nutrition_result = _build_nutrition_response(text, from_text=True)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process this image. Details: {e}",
        )

    return {"product": product, "report": report, "nutrition_analysis": nutrition_result}


@app.post("/scan-product-ai")
async def scan_product_ai(file: UploadFile = File(...)):
    """
    OCR + AI text structuring (step9). OCR reads the image (as above),
    then a fast text AI model (Groq) structures the messy OCR text into
    clean fields - more robust than regex on badly-worded/garbled labels.
    Same rule_engine.py validation step as /scan-product.
    Requires GROQ_API_KEY set as an environment variable.
    """
    from step9_ai_structure_from_text import structure_text_with_ai

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Get a free key at "
                    "https://console.groq.com and set it as an environment variable.",
        )

    image_path = await _validate_and_save_image(file)

    try:
        print(f"[scan-ai] Running OCR on '{image_path}'...")
        text_path, layout_path = ocr_product_image(image_path, OCR_OUTPUT_DIR)
        text = load_text(text_path)

        print(f"[scan-ai] Sending OCR text to Groq for AI structuring...")
        product = structure_text_with_ai(text)
        print(f"[scan-ai] Groq responded. ocr_source={product.get('ocr_source')}")

        compliance_rules = load_json(COMPLIANCE_RULES_PATH)
        report = run_rule_engine(compliance_rules, product)
        print(f"[scan-ai] Result: {report['summary']}")

        nutrition_result = _build_nutrition_response(product, from_text=False)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process this image with AI structuring. Details: {e}",
        )

    return {"product": product, "report": report, "nutrition_analysis": nutrition_result}


@app.post("/scan-product-ai-vision")
async def scan_product_ai_vision(file: UploadFile = File(...)):
    """
    Pure vision AI (step8). AI reads the image PIXELS directly (no
    Tesseract OCR at all) - best for very messy/angled photos where OCR
    itself struggles. Same rule_engine.py validation step as the others.
    Requires GROQ_API_KEY set as an environment variable.
    """
    from step8_vision_extract_product import extract_product_fields_with_vision

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Get a free key at "
                    "https://console.groq.com and set it as an environment variable.",
        )

    image_path = await _validate_and_save_image(file)

    try:
        print(f"[scan-vision] Sending '{image_path}' to Groq vision model...")
        product = extract_product_fields_with_vision(image_path)
        print(f"[scan-vision] Groq responded. ocr_source={product.get('ocr_source')}")

        compliance_rules = load_json(COMPLIANCE_RULES_PATH)
        report = run_rule_engine(compliance_rules, product)
        print(f"[scan-vision] Result: {report['summary']}")

        nutrition_result = _build_nutrition_response(product, from_text=False)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process this image with the vision model. Details: {e}",
        )

    return {"product": product, "report": report, "nutrition_analysis": nutrition_result}


@app.post("/scan-product-hybrid")
async def scan_product_hybrid(file: UploadFile = File(...)):
    """
    HYBRID PRODUCT SCANNER

    Image
      -> Step 6: OCR
      -> Step 7: Regex extraction
      -> Step 8: Vision AI
      -> Step 9: OCR text AI structuring
      -> Step 11: Smart merge
      -> Rule Engine
    """

    image_path = await _validate_and_save_image(file)

    try:
        # =====================================================
        # STEP 6 — OCR
        # =====================================================
        print(f"[hybrid] Running OCR on '{image_path}'...")

        text_path, layout_path = ocr_product_image(
            image_path,
            OCR_OUTPUT_DIR
        )

        text = load_text(text_path)

        print(
            f"[hybrid] OCR complete. "
            f"text={text_path}, layout={layout_path}"
        )

        # =====================================================
        # STEP 7 — REGEX / DETERMINISTIC EXTRACTION
        # =====================================================
        print("[hybrid] Extracting fields via regex...")

        product_regex = build_product_json(
            text_path=text_path,
            layout_path=layout_path,
        )

        # =====================================================
        # STEP 8 — VISION AI
        # =====================================================
        product_vision = None

        if os.environ.get("GROQ_API_KEY"):
            try:
                from step8_vision_extract_product import (
                    extract_product_fields_with_vision
                )

                print(
                    "[hybrid] Sending image to Groq vision model..."
                )

                product_vision = extract_product_fields_with_vision(
                    image_path
                )

                print(
                    "[hybrid] Vision responded successfully."
                )

            except Exception as e:
                print(
                    "[hybrid] Vision extraction failed. "
                    f"Continuing without vision: {e}"
                )
                product_vision = None

        else:
            print(
                "[hybrid] GROQ_API_KEY not set - "
                "skipping AI vision."
            )

        # =====================================================
        # STEP 9 — AI STRUCTURING OF OCR TEXT
        # =====================================================
        product_ai_text = None

        if os.environ.get("GROQ_API_KEY"):
            try:
                print(
                    "[hybrid] Sending OCR text + layout "
                    "to Groq text model..."
                )

                product_ai_text = structure_text_with_ai(
                    text,
                    api_key=os.environ.get("GROQ_API_KEY"),
                    layout_path=layout_path,
                )

                print(
                    "[hybrid] OCR text AI responded successfully."
                )

            except Exception as e:
                print(
                    "[hybrid] OCR text AI failed. "
                    f"Continuing without it: {e}"
                )
                product_ai_text = None

        # =====================================================
        # STEP 11 — SMART HYBRID MERGE
        # =====================================================
        print("[hybrid] Merging extraction results...")

        product = merge_product_data(
            product_vision or {},
            product_regex,
            product_ai_text or {},
        )

        # =====================================================
        # RULE ENGINE
        # =====================================================
        print(
            "[hybrid] Validating against "
            "Legal Metrology rules..."
        )

        compliance_rules = load_json(
            COMPLIANCE_RULES_PATH
        )

        report = run_rule_engine(
            compliance_rules,
            product
        )

        print(
            f"[hybrid] Result: {report['summary']}"
        )

        # =====================================================
        # NUTRITION ANALYSIS
        # =====================================================
        nutrition_result = analyze_nutrition_and_claims(text)

        # Add merged nutrition values if available.
        merged_nutrition = product.get(
            "nutrition",
            {}
        ) or {}

        for key, value in merged_nutrition.items():
            if value:
                nutrition_result["nutrition"][key] = value

        # Add merged claims.
        existing_claim_texts = {
            c["claim_text"].strip().lower()
            for c in nutrition_result["claims"]
        }

        for claim_text in product.get(
            "claims",
            []
        ) or []:

            clean_claim = str(
                claim_text
            ).strip()

            if (
                clean_claim
                and clean_claim.lower()
                not in existing_claim_texts
            ):
                nutrition_result["claims"].append({
                    "claim_text": clean_claim,
                    "category": "unclassified",
                    "status": "NEEDS_VERIFICATION",
                    "evidence": (
                        "Detected during hybrid "
                        "AI/OCR extraction."
                    ),
                })

        nutrition_result[
            "nutrition_fields_not_found"
        ] = [
            k
            for k, v
            in nutrition_result["nutrition"].items()
            if not v
        ]

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not process this image. "
                f"Details: {e}"
            ),
        )

    return {
        "product": product,
        "report": report,
        "nutrition_analysis": nutrition_result,
    }


@app.post("/admin/upload-rules-pdf")
async def upload_rules_pdf(file: UploadFile = File(...)):
    """
    ADMIN SIDE: Upload the official Legal Metrology PDF. Runs
    step1 -> step2 -> step3 -> step4 to (re)build the rule dataset that
    every /scan-product* endpoint validates against.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    os.makedirs(ADMIN_PDF_UPLOAD_DIR, exist_ok=True)
    pdf_path = os.path.join(ADMIN_PDF_UPLOAD_DIR, file.filename)

    contents = await file.read()
    if len(contents) < 1024:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded PDF appears incomplete or empty ({len(contents)} bytes). "
                    f"Please try uploading it again.",
        )
    with open(pdf_path, "wb") as f:
        f.write(contents)

    try:
        convert_pdf_to_images(pdf_path, PDF_PAGES_DIR)
        ocr_all_pages(PDF_PAGES_DIR, OCR_OUTPUT_DIR)

        pages = load_pages(OCR_OUTPUT_DIR)
        rules, skipped, duplicates = extract_rules(pages)
        missing = report_missing_rules(rules)
        save_rule_outputs(rules, RAW_RULES_JSON, RULES_TXT)

        raw_rules_by_number = load_raw_rules(RAW_RULES_JSON)
        compliance_rules, missing_curated = build_compliance_rules(raw_rules_by_number, CURATED_CHECKLIST)
        save_compliance_rules(compliance_rules, COMPLIANCE_RULES_PATH)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process this PDF (it may be corrupt or unreadable). Details: {e}",
        )

    return {
        "pages_processed": len(pages),
        "rules_extracted": len(rules),
        "rule_numbers_found": sorted(r["rule_number"] for r in rules),
        "rule_numbers_missing": missing,
        "duplicate_rule_numbers": duplicates,
        "compliance_checks_built": len(compliance_rules),
        "compliance_rule_numbers_missing": sorted(set(missing_curated)),
    }