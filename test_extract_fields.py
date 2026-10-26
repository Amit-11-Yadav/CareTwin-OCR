"""
test_extract_fields.py — regression tests for extract_fields.py.

Every case here is a real failure that was found while testing CareTwin's OCR
output. Run this after ANY change to extract_fields.py, before eval.py:

    python test_extract_fields.py

It needs no images and no OCR engine, so it takes about a second. eval.py
tells you how accurate the pipeline is on real images; this file tells you
whether a change quietly broke something that used to work.
"""

import sys
from extract_fields import extract_fields

_failures = []


def check(name, text, expect, absent=(), flags_include=(), flags_exclude=()):
    """expect: {field: value} that must be extracted. absent: fields that must NOT be."""
    result = extract_fields(text)
    problems = []
    for field, want in expect.items():
        got = result.get(field)
        if isinstance(want, float) and isinstance(got, (int, float)):
            ok = abs(got - want) < 1e-9
        else:
            ok = got == want
        if not ok:
            problems.append(f"{field}: expected {want!r}, got {got!r}")
    for field in absent:
        if field in result:
            problems.append(f"{field}: should be absent but got {result[field]!r}")
    for flag in flags_include:
        if flag not in result["_flags"]:
            problems.append(f"flag {flag!r} missing from {result['_flags']}")
    for flag in flags_exclude:
        if flag in result["_flags"]:
            problems.append(f"flag {flag!r} should not be present")
    if problems:
        _failures.append((name, problems))


# --- separators between label and value -------------------------------------
check("colon", "BP: 140/90", {"bp_systolic": 140, "bp_diastolic": 90})
check("space BEFORE colon (EasyOCR habit)", "BP : 140/90", {"bp_systolic": 140})
check("space before colon, other field", "Hemoglobin : 13.5", {"hemoglobin": 13.5},
      flags_exclude=["hemoglobin_fuzzy_match"])
check("comma instead of colon", "BP, 120/80", {"bp_systolic": 120, "bp_diastolic": 80})
check("period instead of colon", "Blood Sugar. 142", {"blood_sugar": 142})
check("no punctuation at all", "BP 140/90 Blood Sugar 118", {"bp_systolic": 140, "blood_sugar": 118})
check("stray '!' instead of ')'", "Blood (Urine! Negative", {"urine_blood": "negative"})

# --- labels the OCR misspelled ----------------------------------------------
check("fuzzy cholesterol", "CCholesterat 175", {"cholesterol": 175}, flags_include=["cholesterol_fuzzy_match"])
check("fuzzy hemoglobin", "Hemogiobin: 13.5", {"hemoglobin": 13.5}, flags_include=["hemoglobin_fuzzy_match"])
check("short abbreviations are NOT fuzzy-matched", "wee: 7200", {}, absent=["wbc_count"])

# --- words between label and value ------------------------------------------
check("'WBC count' connector", "Total WBC count 25000", {"wbc_count": 25000.0})
check("'RBC count' connector", "Total RBC count 6.5", {"rbc_count": 6.5})
check("MCHC does not fill MCH", "MCHC: 33", {"mchc": 33.0}, absent=["mch"])

# --- reference ranges must never be read as results -------------------------
check("label followed only by its range", "Neutrophils Low 50 - 62 %", {}, absent=["neutrophils_pct"])
check("result then range -> result wins", "Neutrophils 30 Low 50 - 62 %", {"neutrophils_pct": 30.0})
check("fuzzy label, only a range", "Neutrophiis Low 50 - 62 %", {}, absent=["neutrophils_pct"])
check("fuzzy label, range then real value", "Neutrophiis Low 50 - 62 % 30", {"neutrophils_pct": 30.0})
check("decimal range", "Hemoglobin (Hb) Low 13.00 - 17.00 g/dL", {}, absent=["hemoglobin"])
check("result then decimal range", "Hemoglobin (Hb) 11.50 Low 13.00 - 17.00 g/dL", {"hemoglobin": 11.5})
check("en-dash range", "Total WBC count 4000 \u2013 11000", {}, absent=["wbc_count"])
check("zero is a real value", "Basophils 0", {"basophils_pct": 0.0})

# --- a decimal point misread as a hyphen is NOT a range --------------------
check("33.00 read as '33-00'", "MCHC 33-00 Normal", {"mchc": 33.0})
check("ascending is a range, descending is not", "Mch High 27 - 32", {}, absent=["mch"])

# Real EasyOCR output for the DRLOGY CBC report (from the app's raw-text panel).
# Row labels are garbled, results sit beside reference ranges, and two decimals
# came out as hyphens or spaces.
REAL_EASYOCR = (
    "Hemoalobin (Hb) 11.50 Low 13 00 17,00 Total RBC count 6-50 High 5 50 mivcumm "
    "BLOOD INDICES Packed Cell Volume (PCV) Normal Mcan Corpuscular Volume (MCV) 100 Hormal "
    "101 Mch High 27 - 32 MCHC 33-00 Normal 32. 50 34 50 RDW Smart Paecho 12.00 "
    "Total WBC count T1000 cumtT DIFFERENTIAL WBC COUNT"
)
check("real EasyOCR text, DRLOGY CBC", REAL_EASYOCR,
      {"hemoglobin": 11.5, "mcv": 100.0, "mchc": 33.0},
      absent=["rbc_count", "mch", "wbc_count", "neutrophils_pct", "lymphocytes_pct", "eosinophils_pct"])

# --- the dash in a printed range can be read as other characters ------------
check("range dash read as '='", "Eosinophils Normal 00 = 06", {}, absent=["eosinophils_pct"])
check("range dash read as '~'", "Eosinophils Normal 00 ~ 06", {}, absent=["eosinophils_pct"])
check("range written with 'to'", "Eosinophils Normal 0 to 6", {}, absent=["eosinophils_pct"])
check("decimal read as '=' is still a result", "MCHC 33=00 Normal", {"mchc": 33.0})
check("result before a '=' range still wins", "Eosinophils 2 Normal 00 = 06", {"eosinophils_pct": 2.0})

# Real EasyOCR text for the differential section of the DRLOGY report. The
# patient's results (30, 60, 2, 8, 0) are NOT in it at all - EasyOCR skipped
# the small coloured result column - so the correct output is "not found",
# never a reference-range number.
REAL_EASYOCR_DIFF = (
    'DIFFERENTIAL WBC COUNT "rlogy.co Neutrophils Lowi 50 \u2013 62 Lymphocytes High 20 \u2013 40 '
    "Eosinophils Normal 00 = 06 Monocytes Natmal Basophils Narmal Platelet Cou nt Fedu 20000 "
    "Narmal s0000 410000 cumm"
)
check("real EasyOCR differential section", REAL_EASYOCR_DIFF, {"platelet_count": 20000.0},
      absent=["neutrophils_pct", "lymphocytes_pct", "eosinophils_pct", "monocytes_pct", "basophils_pct"])

# --- different tests that share a word --------------------------------------
check("HbA1c is not hemoglobin", "Glycosylated Hemoglobin 7.2", {"hba1c": 7.2}, absent=["hemoglobin"])
check("misspelled glycosylated", "Glycosylatd Hemoglobin 7.2", {}, absent=["hemoglobin"])
check("Mean Corpuscular Hemoglobin is MCH", "Mean Corpuscular Hemoglobin 29.5", {"mch": 29.5}, absent=["hemoglobin", "mchc"])
check("MCHC long form", "Mean Corpuscular Hemoglobin Concentration 33.1", {"mchc": 33.1}, absent=["hemoglobin", "mch"])
check("real hemoglobin after an HbA1c", "Glycosylated Hemoglobin 6.1 Hemoglobin 13.5", {"hemoglobin": 13.5, "hba1c": 6.1})
check("real hemoglobin before an HbA1c", "Hemoglobin 13.5 Glycosylated Hemoglobin 6.1", {"hemoglobin": 13.5, "hba1c": 6.1})
check("'Fasting: 12 hours' is not a sugar value", "Fasting: 12 hours", {}, absent=["fasting_blood_sugar", "blood_sugar"])
check("fasting sugar", "Fasting Blood Sugar: 95", {"fasting_blood_sugar": 95})
check("PP and random sugar", "PPBS 168 RBS 130", {"post_prandial_blood_sugar": 168, "random_blood_sugar": 130})

# --- values that are out of range are flagged, not trusted ------------------
check("dropped decimal is flagged", "Hemoglobin: 135", {"hemoglobin": 135.0}, flags_include=["hemoglobin_out_of_range"])

# --- dates -------------------------------------------------------------------
check("date parsed", "Date: 10/01/2024", {"date": "2024-01-10"})

# --- medications / diagnosis -------------------------------------------------
meds = extract_fields("Tab Metformin 500mg Tab Amlodipine 5mg")
if meds.get("medications") != [{"name": "Metformin", "dosage": "500mg"}, {"name": "Amlodipine", "dosage": "5mg"}]:
    _failures.append(("medications split into name + dosage", [f"got {meds.get('medications')!r}"]))
check("diagnosis", "Diagnosis: Type 2 Diabetes Mellitus with Hypertension BP: 130/85",
      {"diagnosis": "Type 2 Diabetes Mellitus with Hypertension"}, flags_include=["diagnosis_needs_verification"])
check("nothing invented from ordinary words", "The patient is stable. Case notes reviewed. BP: 118/76",
      {"bp_systolic": 118}, absent=["medications", "diagnosis"])


if __name__ == "__main__":
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for name, problems in _failures:
            print(f"  - {name}")
            for p in problems:
                print(f"      {p}")
        sys.exit(1)
    print("all extraction regression tests passed")