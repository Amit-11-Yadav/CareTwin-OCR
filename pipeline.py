"""
pipeline.py — the single function the backend track calls.

image path in -> structured JSON out.
"""

from preprocess import preprocess_image, preprocess_image_light
from ocr_engine import run_ocr
from extract_fields import extract_fields
from quality_check import assess_quality
import cv2
import difflib
import re

CONFIDENCE_THRESHOLD = 60.0
RAW_GOOD_ENOUGH = 75.0  # if the untouched image already scores above this, skip preprocessing

# Tesseract is a classical engine and benefits from hard binarization/contrast
# correction. EasyOCR/PaddleOCR are deep-learning engines that do their own
# internal preprocessing — feeding them a binarized image throws away
# grayscale information their networks rely on and measurably hurts results.
# So each engine gets a different preprocessing function, not the same one.
_PREPROCESSORS = {
    "tesseract": preprocess_image,
    "easyocr": preprocess_image_light,
    "paddleocr": preprocess_image_light,
}


# --- Completeness check -----------------------------------------------------
# A printed report can pass every confidence check and still silently lose a
# value (e.g. OCR reads "BP," instead of "BP:", or misspells "Cholesterol").
# If the raw text still MENTIONS a field but nothing was extracted for it,
# that is a partial extraction and must be reviewed, not called trustworthy.
# key: (extracted field that satisfies it, label words that mention it)
_EXPECTED_FIELDS = {
    "bp": ("bp_systolic", ["bp", "pressure"]),
    "blood_sugar": ("blood_sugar", ["sugar", "glucose", "rbs", "fbs"]),
    "hemoglobin": ("hemoglobin", ["hemoglobin", "haemoglobin", "hb"]),
    "cholesterol": ("cholesterol", ["cholesterol", "ldl"]),
    "date": ("date", ["date"]),
}


def _mentions(token: str, keyword: str) -> bool:
    """Short labels (bp, hb, rbs, fbs) must match exactly; longer ones tolerate OCR misspellings."""
    if len(keyword) <= 3:
        return token == keyword
    cutoff = 0.65 if len(keyword) >= 8 else 0.8
    return difflib.SequenceMatcher(None, token, keyword).ratio() >= cutoff


def find_unparsed_fields(raw_text: str, fields: dict) -> list:
    """Names of fields the text mentions but for which no value was extracted."""
    tokens = re.findall(r"[a-z]+", raw_text.lower())
    missing = []
    for name, (extracted_key, keywords) in _EXPECTED_FIELDS.items():
        if extracted_key in fields:
            continue
        if any(_mentions(t, kw) for t in tokens for kw in keywords):
            missing.append(name)
    return missing


def process_document(image_path: str, engine: str = "tesseract", quality_gate: bool = True) -> dict:
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Gate: blurry, faint, tiny, or handwritten documents are NOT sent to OCR.
    # They are archived as-is in the patient's history and marked "not machine-read",
    # because OCR on them produces confident-looking garbage. Set quality_gate=False
    # to bypass (e.g. when benchmarking engines on hard images).
    if quality_gate:
        quality = assess_quality(raw_img)
        if not quality.passed:
            return {
                "status": "stored_only",
                "raw_text": "",
                "ocr_confidence": 0.0,
                "fields": {},
                "flags": list(quality.reasons),
                "needs_review": True,
                "engine": "none",
                "quality": {"doc_type": quality.doc_type, **quality.metrics},
            }
    else:
        quality = None

    raw_result = run_ocr(raw_img, engine=engine)

    if raw_result.mean_confidence >= RAW_GOOD_ENOUGH:
        ocr_result = raw_result
    else:
        preprocess_fn = _PREPROCESSORS.get(engine, preprocess_image_light)
        cleaned_img = preprocess_fn(image_path)
        pre_result = run_ocr(cleaned_img, engine=engine)
        ocr_result = pre_result if pre_result.mean_confidence > raw_result.mean_confidence else raw_result

    fields = extract_fields(ocr_result.text)
    # bool(...) guards against numpy bool types leaking in from an engine's
    # confidence score (numpy.bool_ isn't JSON-serializable, unlike Python's
    # native bool) — belt-and-suspenders alongside the fix in ocr_engine.py.
    unparsed = find_unparsed_fields(ocr_result.text, fields)
    all_flags = list(fields["_flags"]) + [f"unparsed_{name}" for name in unparsed]
    needs_review = bool(
        fields["_needs_review"] or unparsed or ocr_result.mean_confidence < CONFIDENCE_THRESHOLD
    )

    return {
        "status": "processed",
        "raw_text": ocr_result.text,
        "ocr_confidence": round(float(ocr_result.mean_confidence), 1),
        "fields": {k: v for k, v in fields.items() if not k.startswith("_")},
        "flags": all_flags,
        "needs_review": needs_review,
        "engine": ocr_result.engine,
        "quality": ({"doc_type": quality.doc_type, **quality.metrics} if quality else None),
    }


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run the OCR pipeline on an image.")
    parser.add_argument("image_path", help="Path to the document image")
    parser.add_argument(
        "--engine",
        default="tesseract",
        choices=["tesseract", "easyocr", "paddleocr"],
        help="OCR engine to use (default: tesseract)",
    )
    args = parser.parse_args()

    print(json.dumps(process_document(args.image_path, engine=args.engine), indent=2))