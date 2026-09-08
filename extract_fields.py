"""
extract_fields.py — Week 4 of the OCR track (the NLP half).

Rule-based extraction: regex + a keyword dictionary. Chosen over a general
NER model because the field set is small and fixed (BP, sugar, cholesterol,
Hb, medications, date) — for a bounded vocabulary like this, rules are more
predictable and easier to debug than an ML model, and there's no training
data available to fine-tune one anyway.

Output shape matches the schema the ML track consumes downstream:
    {"bp_systolic": 140, "bp_diastolic": 90, "date": "2024-03-15", ...}
"""

import re
from datetime import datetime


# --- Common OCR misreads, fixed before field extraction runs --------------
# Tesseract frequently confuses these characters in numeric contexts.
_OCR_TYPO_FIXES = [
    (r"(?<=\d)O(?=\d)", "0"),   # letter O misread inside a number
    (r"(?<=\d)l(?=\d)", "1"),   # lowercase l misread inside a number
    (r"(?<=\d)I(?=\d)", "1"),   # capital I misread inside a number
]


def clean_text(raw_text: str) -> str:
    text = raw_text
    for pattern, replacement in _OCR_TYPO_FIXES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Field patterns ---------------------------------------------------------
# Each pattern is deliberately permissive about spacing/punctuation since
# OCR output is messy, but anchored to a keyword so we don't grab random
# numbers that aren't actually the field in question.

_PATTERNS = {
    "bp": re.compile(
        r"\b(?:BP|Blood\s*Pressure)\b\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
        re.IGNORECASE,
    ),
    "blood_sugar": re.compile(
        r"\b(?:Blood\s*Sugar|Glucose|RBS|FBS)\b\s*[:\-]?\s*(\d{2,3})",
        re.IGNORECASE,
    ),
    "hemoglobin": re.compile(
        r"\b(?:Hb|Hemoglobin|Haemoglobin)\b\s*[:\-]?\s*(\d{1,2}\.?\d?)",
        re.IGNORECASE,
    ),
    "cholesterol": re.compile(
        r"\b(?:Cholesterol|LDL|Total\s*Cholesterol)\b\s*[:\-]?\s*(\d{2,3})",
        re.IGNORECASE,
    ),
    "date": re.compile(
        r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"
    ),
}

# Range-based sanity checks — catches silent OCR misreads (e.g. "140" read
# as "40"). A value outside these bounds is almost certainly a misread,
# not a real clinical value, so we flag it rather than trust it blindly.
_VALID_RANGES = {
    "bp_systolic": (60, 250),
    "bp_diastolic": (40, 150),
    "blood_sugar": (40, 500),
    "hemoglobin": (3.0, 20.0),
    "cholesterol": (80, 400),
}


def _in_range(field: str, value: float) -> bool:
    lo, hi = _VALID_RANGES.get(field, (float("-inf"), float("inf")))
    return lo <= value <= hi


def _parse_date(raw: str) -> str | None:
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None  # unparseable — leave it out rather than guess


def extract_fields(raw_text: str) -> dict:
    """
    Returns a dict of extracted fields plus a per-field `_flags` list noting
    anything that failed a sanity check — the UI should surface these as
    "please verify" rather than silently trusting them.
    """
    text = clean_text(raw_text)
    result: dict = {}
    flags: list[str] = []

    bp_match = _PATTERNS["bp"].search(text)
    if bp_match:
        systolic, diastolic = int(bp_match.group(1)), int(bp_match.group(2))
        result["bp_systolic"] = systolic
        result["bp_diastolic"] = diastolic
        if not _in_range("bp_systolic", systolic) or not _in_range("bp_diastolic", diastolic):
            flags.append("bp_out_of_range")

    for field, key in [("blood_sugar", "blood_sugar"), ("hemoglobin", "hemoglobin"),
                        ("cholesterol", "cholesterol")]:
        m = _PATTERNS[field].search(text)
        if m:
            value = float(m.group(1))
            result[key] = value
            if not _in_range(key, value):
                flags.append(f"{key}_out_of_range")

    date_match = _PATTERNS["date"].search(text)
    if date_match:
        parsed = _parse_date(date_match.group(1))
        if parsed:
            result["date"] = parsed
        else:
            flags.append("date_unparseable")

    result["_flags"] = flags
    result["_needs_review"] = len(flags) > 0
    return result


if __name__ == "__main__":
    sample = "Patient Report BP: 14O/90 Blood Sugar: 118 Date: 15/O3/2024"
    print(extract_fields(sample))
