"""
pipeline.py — the single function the backend track calls.

image path in -> structured JSON out.
"""

from preprocess import preprocess_image, preprocess_image_light
from ocr_engine import run_ocr
from extract_fields import extract_fields
from quality_check import assess_quality, median_char_height
import cv2
import difflib
import os
import re
import tempfile

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


# --- Multi-scale OCR ---------------------------------------------------------
# OCR engines read text best when characters are roughly 20-40 px tall. Lab
# reports photographed or exported at normal page size are often smaller, and
# the result is dropped decimal points and 6/8 digit swaps. Measured on the
# test set with Tesseract: 10 px text -> 5/28 fields right, the SAME pixels
# enlarged 2x -> 28/28; the real DRLOGY CBC report went from 3/13 to 12/13.
#
# But the effect is bumpy, not smooth (that report scored 0/13 at 1.5x and 2x,
# 12/13 at 2.5x), so one fixed scale is fragile. Instead several scales are
# tried and the pipeline keeps the one that yields the most CLEAN fields -
# extracted with no flag raised against them. Mean OCR confidence is not used:
# it rose to 82 at the scales that extracted almost nothing.
ALREADY_LARGE_PX = 20.0               # text this tall needs no enlarging
UPSCALE_TARGET_HEIGHTS = (24, 32, 40)  # character heights (px) to aim for
MAX_UPSCALE = 4.0
MAX_IMAGE_SIDE = 5000                  # never build an image bigger than this (bounds run time)

# Only Tesseract is enabled by default. EasyOCR has its own internal resizing
# and enlarging has not been measured for it - turn it on per call with
# process_document(..., upscale=True) or `eval.py --upscale on` and compare.
UPSCALE_BY_DEFAULT = {"tesseract": True, "easyocr": False, "paddleocr": False}


def candidate_scales(raw_img) -> list:
    """Scales worth trying for this image, always starting with 1.0 (untouched)."""
    scales = [1.0]
    height = median_char_height(raw_img)
    if height <= 0 or height >= ALREADY_LARGE_PX:
        return scales
    longest = max(raw_img.shape[:2])
    for target in UPSCALE_TARGET_HEIGHTS:
        s = min(round(target / height * 2) / 2, MAX_UPSCALE)   # nearest 0.5
        if s > 1.25 and s not in scales and longest * s <= MAX_IMAGE_SIDE:
            scales.append(s)
    return scales


def _clean_field_count(extracted: dict) -> tuple:
    """(fields with no flag raised against them, all fields) - the scale-selection score."""
    flags = extracted["_flags"]
    fields = [k for k in extracted if not k.startswith("_")]
    clean = [
        f for f in fields
        if not any(x.startswith(("bp" if f.startswith("bp_") else f) + "_") for x in flags)
    ]
    return len(clean), len(fields)


def _ocr_once(image_path: str, raw_img, engine: str):
    """One OCR pass: raw image first, preprocessed only if the raw read scored poorly."""
    raw_result = run_ocr(raw_img, engine=engine)
    if raw_result.mean_confidence >= RAW_GOOD_ENOUGH:
        return raw_result
    preprocess_fn = _PREPROCESSORS.get(engine, preprocess_image_light)
    pre_result = run_ocr(preprocess_fn(image_path), engine=engine)
    return pre_result if pre_result.mean_confidence > raw_result.mean_confidence else raw_result


def _best_ocr(image_path: str, raw_img, engine: str, use_upscale: bool):
    """Try each candidate scale; return (ocr_result, extracted_fields, scale_used)."""
    scales = candidate_scales(raw_img) if use_upscale else [1.0]
    best = None
    with tempfile.TemporaryDirectory() as tmp:
        for scale in scales:
            if scale == 1.0:
                path, img = image_path, raw_img
            else:
                img = cv2.resize(raw_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                path = os.path.join(tmp, f"scaled_{scale}.png")
                cv2.imwrite(path, img)
            result = _ocr_once(path, img, engine)
            extracted = extract_fields(result.text)
            score = _clean_field_count(extracted)
            if best is None or score > best[0]:     # strict > : ties keep the smaller scale
                best = (score, result, extracted, scale)
    return best[1], best[2], best[3]


def process_document(image_path: str, engine: str = "tesseract", quality_gate: bool = True,
                     upscale=None) -> dict:
    """upscale: None = engine default (see UPSCALE_BY_DEFAULT), True/False to force."""
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

    use_upscale = UPSCALE_BY_DEFAULT.get(engine, False) if upscale is None else bool(upscale)
    ocr_result, fields, scale_used = _best_ocr(image_path, raw_img, engine, use_upscale)
    # bool(...) guards against numpy bool types leaking in from an engine's
    # confidence score (numpy.bool_ isn't JSON-serializable, unlike Python's
    # native bool) — belt-and-suspenders alongside the fix in ocr_engine.py.
    unparsed = find_unparsed_fields(ocr_result.text, fields)
    all_flags = list(fields["_flags"]) + [f"unparsed_{name}" for name in unparsed]
    # A document that was read with high confidence but yielded NO recognised
    # value must not look trustworthy: "nothing found" is not "all checks
    # passed" (e.g. a clean letter with no lab values scored 96% confidence).
    nothing_found = not any(not k.startswith("_") for k in fields)
    if nothing_found:
        all_flags.append("no_fields_found")
    needs_review = bool(
        fields["_needs_review"] or unparsed or nothing_found
        or ocr_result.mean_confidence < CONFIDENCE_THRESHOLD
    )

    return {
        "status": "processed",
        "raw_text": ocr_result.text,
        "ocr_confidence": round(float(ocr_result.mean_confidence), 1),
        "fields": {k: v for k, v in fields.items() if not k.startswith("_")},
        "flags": all_flags,
        "needs_review": needs_review,
        "engine": ocr_result.engine,
        "ocr_scale": scale_used,
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
    parser.add_argument(
        "--upscale", choices=["auto", "on", "off"], default="auto",
        help="enlarge small text before OCR (auto = on for tesseract, off for easyocr)",
    )
    args = parser.parse_args()

    upscale = {"auto": None, "on": True, "off": False}[args.upscale]
    print(json.dumps(process_document(args.image_path, engine=args.engine, upscale=upscale), indent=2))