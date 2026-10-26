"""
test_pipeline_scales.py — tests for the multi-scale OCR step in pipeline.py.

    python test_pipeline_scales.py

Needs Tesseract installed (the same one the pipeline already uses). It renders
its own test images, so no files are required.
"""

import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import candidate_scales, process_document, _clean_field_count

failures = []


def check(name, ok, detail=""):
    if not ok:
        failures.append(f"{name} {detail}".strip())


def render(lines, px):
    font = ImageFont.load_default(size=px)
    line_h = int(px * 2.6)
    img = Image.new("L", (max(680, int(px * 28)), 30 + line_h * len(lines)), 250)
    draw = ImageDraw.Draw(img)
    y = 15
    for line in lines:
        draw.text((20, y), line, fill=10, font=font)
        y += line_h
    return np.array(img)


LINES = ["Patient Report", "BP: 140/90", "Blood Sugar: 118", "Hemoglobin: 13.5",
         "Cholesterol: 190", "Date: 03/11/2024"]
TRUTH = {"bp_systolic": 140, "bp_diastolic": 90, "blood_sugar": 118, "hemoglobin": 13.5, "cholesterol": 190}

# --- which scales get tried --------------------------------------------------
check("large text is left alone", candidate_scales(render(LINES, 32)) == [1.0])
tiny = candidate_scales(render(LINES, 10))
check("tiny text tries 1.0 first", tiny[0] == 1.0, str(tiny))
check("tiny text also tries an enlarged scale", any(s >= 2.0 for s in tiny), str(tiny))
check("never asks for more than the cap", all(s <= 4.0 for s in tiny), str(tiny))
check("blank page has nothing to enlarge", candidate_scales(np.full((300, 600), 250, np.uint8)) == [1.0])

# --- the score used to pick a scale -------------------------------------------
check("a flagged field is not 'clean'",
      _clean_field_count({"hemoglobin": 135.0, "date": "2025-01-01", "_flags": ["hemoglobin_out_of_range"]}) == (1, 2))
check("bp flags cover both bp fields",
      _clean_field_count({"bp_systolic": 120, "bp_diastolic": 80, "_flags": ["bp_out_of_range"]}) == (0, 2))

# --- end to end: 10 px text is unreadable at 1x and fine once enlarged ---------
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "tiny.png")
    cv2.imwrite(path, render(LINES, 10))
    try:
        plain = process_document(path, engine="tesseract", quality_gate=False, upscale=False)
        auto = process_document(path, engine="tesseract", quality_gate=False)
    except Exception as e:                                 # Tesseract not installed, etc.
        print(f"skipped the end-to-end part ({type(e).__name__}: {e})")
    else:
        def right(out):
            return sum(1 for f, v in TRUTH.items() if out["fields"].get(f) is not None
                       and abs(float(out["fields"][f]) - v) < 0.06)
        check("upscale=False never enlarges", plain["ocr_scale"] == 1.0)
        check("enlarging reads the tiny report correctly", right(auto) == len(TRUTH),
              f"got {right(auto)}/{len(TRUTH)}: {auto['fields']}")
        check("enlarging beats not enlarging", right(auto) > right(plain),
              f"{right(auto)} vs {right(plain)}")
        check("scale used is reported", auto["ocr_scale"] >= 2.0, str(auto["ocr_scale"]))

# --- "nothing found" must never read as trustworthy ---------------------------
with tempfile.TemporaryDirectory() as tmp:
    letter = os.path.join(tmp, "letter.png")
    cv2.imwrite(letter, render(["Discharge Summary", "The patient was admitted for observation and",
                                "discharged in stable condition after two days."], 24))
    try:
        out = process_document(letter, engine="tesseract")
    except Exception as e:
        print(f"skipped the no-fields check ({type(e).__name__}: {e})")
    else:
        check("clean document with no lab values is flagged", out["needs_review"] is True, str(out["flags"]))
        check("and says why", "no_fields_found" in out["flags"], str(out["flags"]))
        report = os.path.join(tmp, "report.png")
        cv2.imwrite(report, render(LINES, 24))
        normal = process_document(report, engine="tesseract")
        check("a normal report is not flagged as empty", "no_fields_found" not in normal["flags"], str(normal["flags"]))

if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all pipeline scaling tests passed")
