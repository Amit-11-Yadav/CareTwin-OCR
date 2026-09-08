"""
pipeline.py — the single function the backend track calls.

image path in -> structured JSON out.
"""

from preprocess import preprocess_image, preprocess_image_light
from ocr_engine import run_ocr
from extract_fields import extract_fields
import cv2

CONFIDENCE_THRESHOLD = 60.0
RAW_GOOD_ENOUGH = 45.0  # if the untouched image already scores above this, skip preprocessing

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


def process_document(image_path: str, engine: str = "tesseract") -> dict:
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    raw_result = run_ocr(raw_img, engine=engine)
    print(f"[DEBUG] Raw confidence: {raw_result.mean_confidence}")  # ADD THIS

    if raw_result.mean_confidence >= RAW_GOOD_ENOUGH:
        ocr_result = raw_result
    else:
        cleaned_img = preprocess_image(image_path)
        pre_result = run_ocr(cleaned_img, engine=engine)
        print(f"[DEBUG] Preprocessed confidence: {pre_result.mean_confidence}")  # ADD THIS
        ocr_result = pre_result if pre_result.mean_confidence > raw_result.mean_confidence else raw_result
    

    fields = extract_fields(ocr_result.text)
    # bool(...) guards against numpy bool types leaking in from an engine's
    # confidence score (numpy.bool_ isn't JSON-serializable, unlike Python's
    # native bool) — belt-and-suspenders alongside the fix in ocr_engine.py.
    needs_review = bool(fields["_needs_review"] or ocr_result.mean_confidence < CONFIDENCE_THRESHOLD)

    return {
        "raw_text": ocr_result.text,
        "ocr_confidence": round(float(ocr_result.mean_confidence), 1),
        "fields": {k: v for k, v in fields.items() if not k.startswith("_")},
        "flags": fields["_flags"],
        "needs_review": needs_review,
        "engine": ocr_result.engine,
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