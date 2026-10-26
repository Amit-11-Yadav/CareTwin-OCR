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
import difflib
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
#
# Separator now also tolerates a comma: real eval runs showed Tesseract/EasyOCR
# reading "BP:" as "BP," fairly often, and the old [:\-]? never matched that.

_SEP = r"[^\w\s]{0,2}"  # 0-2 stray punctuation chars — covers :/-/,/. and one-off OCR noise like "!"


        # A real lab report often puts a connector word between the test
        # name and its value — "Total WBC count 25000", "Hemoglobin level:
        # 11.5" — which plain punctuation tolerance doesn't cover. Allow ONE
        # of a small, specific set of words here, not any word: letting
        # through an arbitrary word would risk matching a number far from
        # the one the keyword actually refers to.
_CONNECTOR = r"(?:\s*(?:count|level|value))?"


def _pattern(keywords: str, value_regex: str) -> re.Pattern:
    # Only a LEADING word boundary — a trailing \b breaks whenever a keyword
    # can legitimately end in punctuation (e.g. "Blood (Urine)" ending in a
    # literal ")"), since \b can't fire between two non-word characters.
    #
    # \s* before the separator is required: OCR (EasyOCR especially) often
    # emits "BP : 140/90" with a space ahead of the colon.
    #
    # Reference ranges ("50 - 62") are rejected in code by _is_range_bound(),
    # not here — telling a range from a misread decimal ("33-00" for 33.00)
    # needs a numeric comparison, which a regex can't do.
    return re.compile(
        rf"\b(?:{keywords}){_CONNECTOR}\s*{_SEP}\s*({value_regex})",
        re.IGNORECASE,
    )


# --- Single-value numeric fields --------------------------------------------
# One table drives matching, range-checking, AND the fuzzy-keyword fallback,
# so adding a new lab value is one line here rather than touching four places.
#
# Each entry: key -> (compiled pattern, valid_range, fuzzy_keywords or None)
#
# fuzzy_keywords is None for short abbreviations (Hb, WBC, RBC, MCV, MCH,
# MCHC, SGOT, SGPT, ALP, ALT, AST, PLT, PCV, TLC) — a fuzzy match on 2-4
# letters is too likely to hit an unrelated word by chance and silently
# attach a wrong number to the wrong field. Only long, distinctive words get
# a fuzzy fallback.
_NUMERIC_FIELDS = {
    "blood_sugar": (
        _pattern(r"Blood\s*Sugar|Glucose|RBS|FBS", r"\d{2,3}"),
        (40, 500), ["sugar", "glucose"],
    ),
    # Specific sugar tests. A report that says "Fasting Blood Sugar 95" will
    # fill BOTH fasting_blood_sugar and the generic blood_sugar above (the
    # generic pattern also sees the words "Blood Sugar") — downstream code
    # should prefer the specific field when it is present. No fuzzy fallback
    # here: qualifier words like "fasting" are ordinary English and would
    # fuzzy-match unrelated text such as "Fasting: 12 hours".
    "fasting_blood_sugar": (
        _pattern(r"Fasting\s*Blood\s*Sugar|Fasting\s*Glucose|FBS|FBG", r"\d{2,3}"),
        (40, 500), None,
    ),
    "post_prandial_blood_sugar": (
        _pattern(r"Post\s*Prandial\s*(?:Blood\s*)?(?:Sugar|Glucose)|PPBS|PP2BS|PPBG|PP\s*Blood\s*Sugar", r"\d{2,3}"),
        (40, 600), None,
    ),
    "random_blood_sugar": (
        _pattern(r"Random\s*Blood\s*Sugar|Random\s*Glucose|RBS|RBG", r"\d{2,3}"),
        (40, 600), None,
    ),
    "hba1c": (
        _pattern(r"HbA1c|Hb\s*A1c|Glycosylated\s*Hemoglobin|Glycated\s*Hemoglobin|A1c", r"\d{1,2}\.?\d?"),
        (3.0, 15.0), ["glycosylated"],
    ),
    "hemoglobin": (
        # "Glycosylated Hemoglobin 7.2" (HbA1c) and "Mean Corpuscular
        # Hemoglobin 29.5" (MCH) must NOT be read as plain hemoglobin — an
        # HbA1c of 7.2% would otherwise show up as a hemoglobin of 7.2 g/dL,
        # i.e. severe anemia. That is enforced in extract_fields() via
        # _FUZZY_EXCLUDE_PRECEDING, which tolerates OCR misspellings of the
        # preceding word (a regex lookbehind only catches exact spelling).
        _pattern(r"Hb|Hemoglobin|Haemoglobin", r"\d{1,2}\.?\d?"),
        (3.0, 20.0), ["hemoglobin", "haemoglobin"],
    ),
    "cholesterol": (
        _pattern(r"Cholesterol|LDL|Total\s*Cholesterol", r"\d{2,3}"),
        (80, 400), ["cholesterol"],
    ),

    # --- CBC: Complete Blood Count ---
    "wbc_count": (
        _pattern(r"WBC|Total\s*Leukocyte\s*Count|TLC", r"\d{3,6}"),
        (2000, 20000), None,
    ),
    "rbc_count": (
        _pattern(r"RBC|Red\s*Blood\s*Cell\s*Count", r"\d\.\d{1,2}"),
        (2.5, 7.0), None,
    ),
    "platelet_count": (
        _pattern(r"Platelet(?:\s*Count)?|PLT", r"\d{4,6}"),
        (50000, 600000), ["platelet"],
    ),
    "hematocrit": (
        _pattern(r"Hematocrit|Haematocrit|PCV", r"\d{2,3}\.?\d?"),
        (25, 60), ["hematocrit", "haematocrit"],
    ),
    "mcv": (_pattern(r"MCV", r"\d{2,3}\.?\d?"), (60, 110), None),
    "mch": (
        _pattern(r"MCH(?!C)|Mean\s*Corpuscular\s*Hemoglobin(?!\s*Conc)", r"\d{2}\.?\d?"),
        (18, 40), None,
    ),
    "mchc": (
        _pattern(r"MCHC|Mean\s*Corpuscular\s*Hemoglobin\s*Conc\w*", r"\d{2}\.?\d?"),
        (28, 38), None,
    ),

    # --- CBC differential WBC count (percentages) ---
    # Ranges here are wide plausibility bounds for catching misreads, not
    # clinical reference ranges. The labels are long, distinctive words, so
    # they get a fuzzy fallback.
    "neutrophils_pct": (_pattern(r"Neutrophils?", r"\d{1,2}\.?\d?"), (10, 95), ["neutrophils"]),
    "lymphocytes_pct": (_pattern(r"Lymphocytes?", r"\d{1,2}\.?\d?"), (5, 80), ["lymphocytes"]),
    "eosinophils_pct": (_pattern(r"Eosinophils?", r"\d{1,2}\.?\d?"), (0, 30), ["eosinophils"]),
    "monocytes_pct": (_pattern(r"Monocytes?", r"\d{1,2}\.?\d?"), (0, 25), ["monocytes"]),
    "basophils_pct": (_pattern(r"Basophils?", r"\d{1,2}\.?\d?"), (0, 10), ["basophils"]),

    # --- LFT: Liver Function Test ---
    "sgot": (_pattern(r"SGOT|AST", r"\d{1,4}"), (5, 200), None),
    "sgpt": (_pattern(r"SGPT|ALT", r"\d{1,4}"), (5, 200), None),
    "bilirubin_total": (
        _pattern(r"Total\s*Bilirubin|Bilirubin\s*Total|Bilirubin", r"\d{1,2}\.?\d{0,2}"),
        (0.1, 5.0), ["bilirubin"],
    ),
    "alp": (
        _pattern(r"ALP|Alkaline\s*Phosphatase", r"\d{2,4}"),
        (30, 400), ["phosphatase"],
    ),
    "total_protein": (
        _pattern(r"Total\s*Protein", r"\d{1,2}\.?\d?"),
        (4.0, 9.0), ["protein"],
    ),
    "albumin": (_pattern(r"Albumin", r"\d{1,2}\.?\d?"), (2.0, 6.0), ["albumin"]),

    # --- Urine test (numeric parameters only — see categorical ones below) ---
    "urine_ph": (_pattern(r"Urine\s*pH|pH", r"\d\.?\d?"), (4.5, 8.5), None),
    "urine_specific_gravity": (
        _pattern(r"Specific\s*Gravity|Sp\.?\s*Gravity", r"1\.\d{3}"),
        (1.000, 1.035), ["gravity"],
    ),
}

# Words that, when they directly precede a label, make it a DIFFERENT test.
# The exact patterns handle this with regex lookbehinds; the fuzzy fallback
# needs the same protection or it would walk straight past them.
_FUZZY_EXCLUDE_PRECEDING = {
    "hemoglobin": ("glycosylated", "glycated", "corpuscular"),
}

_FUZZY_CUTOFF = 0.75          # e.g. "cholesterat" vs "cholesterol" = 0.82
_FUZZY_SEARCH_WINDOW = 20     # characters after the label to look for a number
_WORD_RE = re.compile(r"[A-Za-z]+")


def _is_excluded_word(word: str, field: str) -> bool:
    """True if `word` (fuzzily) is one that turns this field's label into a different test."""
    return any(
        difflib.SequenceMatcher(None, word.lower(), ex).ratio() >= _FUZZY_CUTOFF
        for ex in _FUZZY_EXCLUDE_PRECEDING.get(field, ())
    )


def _preceded_by_excluded(text: str, start: int, field: str) -> bool:
    if field not in _FUZZY_EXCLUDE_PRECEDING:
        return False
    before = _WORD_RE.findall(text[:start])
    return bool(before) and _is_excluded_word(before[-1], field)


_NUM = r"\d+(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
# What can sit between the two ends of a printed range. OCR reads the dash in
# "00 - 06" as many things; "=" was seen in a real EasyOCR run ("00 = 06").
# Wrongly treating a separator as a range is safe (the value is simply not
# used); the ascending-numbers rule below is what protects real results.
_RANGE_SEP = r"(?:[-\u2013\u2014\u2212=~_]|\bto\b)"
_DASH_THEN_NUM = re.compile(rf"\s*{_RANGE_SEP}\s*({_NUM})", re.IGNORECASE)
_NUM_THEN_DASH = re.compile(rf"({_NUM})\s*{_RANGE_SEP}\s*$", re.IGNORECASE)


def _is_range_bound(text: str, start: int) -> bool:
    """
    True if the number starting at text[start] is one end of a reference
    range like "50 - 62" or "13.00 - 17.00".

    Lab tables print a range beside every result, so when OCR misses the real
    result the nearest number is a range bound — a plausible-looking WRONG
    value, which is worse than no value.

    A real range always ASCENDS (low - high). That is what separates it from
    a decimal point OCR misread as a hyphen: "33-00" is 33.00, and 00 < 33.
    """
    m = _NUM_RE.match(text, start)
    if not m:
        return False
    value = float(m.group())
    after = _DASH_THEN_NUM.match(text, m.end())
    if after and float(after.group(1)) > value:
        return True    # lower bound of "value - higher"
    before = _NUM_THEN_DASH.search(text[:start])
    if before and float(before.group(1)) < value:
        return True    # upper bound of "lower - value"
    return False


def _fuzzy_find(text: str, field: str) -> str | None:
    """Best-effort value lookup when the exact keyword regex found nothing."""
    keywords = _NUMERIC_FIELDS[field][2]
    if not keywords:
        return None
    for match in _WORD_RE.finditer(text):
        token = match.group().lower()
        if len(token) < 5:
            continue  # too short for a safe fuzzy comparison
        if _preceded_by_excluded(text, match.start(), field):
            continue  # e.g. the "Hemoglobin" in "Glycosylated Hemoglobin" is a different test
        for kw in keywords:
            if difflib.SequenceMatcher(None, token, kw).ratio() >= _FUZZY_CUTOFF:
                start = match.end()
                window = text[start: start + _FUZZY_SEARCH_WINDOW]
                for num_match in re.finditer(r"\d+\.?\d*", window):
                    if _is_range_bound(text, start + num_match.start()):
                        continue  # a reference-range number, not the patient's result
                    return num_match.group()
    return None


# --- Categorical urine fields -------------------------------------------------
# Urinalysis strip results (protein, glucose, blood) are reported as words or
# grades, not numbers — "Nil", "Trace", "+1".."+4", "Positive"/"Negative" —
# so they get their own small matcher instead of forcing them into the
# numeric table above.
_CATEGORICAL_VALUES = ["nil", "trace", "absent", "present", "negative", "positive", "+1", "+2", "+3", "+4"]
_CATEGORICAL_VALUE_RE = "|".join(re.escape(v) for v in _CATEGORICAL_VALUES)

_CATEGORICAL_FIELDS = {
    "urine_protein": _pattern(r"Urine\s*Protein|Protein\s*\(?Urine\)?|Albumin\s*\(?Urine\)?", _CATEGORICAL_VALUE_RE),
    "urine_glucose": _pattern(r"Urine\s*(?:Glucose|Sugar)|Glucose\s*\(?Urine\)?", _CATEGORICAL_VALUE_RE),
    "urine_blood": _pattern(r"(?:Occult\s*)?Blood\s*\(?Urine\)?", _CATEGORICAL_VALUE_RE),
}
# Urine pus cells, epithelial cells, casts and crystals are deliberately left
# out: real reports report these as free-form ranges ("2-4/hpf", "occasional"),
# which is too variable to regex reliably — better to leave them for manual
# entry than to silently produce a wrong structured value.


# BP and date have their own shape (two values / a date format) so they stay
# separate from the single-value table above.
_BP_PATTERN = _pattern(r"BP|Blood\s*Pressure", r"\d{2,3}\s*/\s*\d{2,3}")
_BP_SPLIT = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")
_DATE_PATTERN = re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b")

# Range-based sanity checks — catches silent OCR misreads (e.g. "140" read
# as "40"). A value outside these bounds is almost certainly a misread, not a
# real clinical value, so we flag it rather than trust it blindly. Built from
# _NUMERIC_FIELDS so every field's range lives in exactly one place.
_VALID_RANGES = {
    "bp_systolic": (60, 250),
    "bp_diastolic": (40, 150),
    **{key: spec[1] for key, spec in _NUMERIC_FIELDS.items()},
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


# --- Medications -------------------------------------------------------------
# OCR output has no line breaks by the time it reaches this function (both
# engines join recognised words with single spaces — see ocr_engine.py), so
# medications can't be parsed "one per line" the way a human reads a
# prescription. Instead, every dosage-form abbreviation (Tab, Cap, Inj, or the
# single-letter "T." / "C." style common on Indian prescriptions) is treated
# as the start of one medication mention, and the drug name + optional dosage
# immediately after it is captured.
#
# This is inherently approximate: it can't verify a name is a real drug, and
# a stray word right after "Tab" could be mistaken for one. That is exactly
# why every match is flagged for human confirmation rather than trusted.
_MED_PREFIX = r"(?:Tab(?:let)?|Cap(?:sule)?|Inj(?:ection)?|Syp|Syrup|Oint(?:ment)?|Susp(?:ension)?|Drops?|[TCIS])\.?"
_MED_PATTERN = re.compile(
    rf"\b{_MED_PREFIX}\s+([A-Za-z][A-Za-z\-]{{1,20}})\s*(\d+\.?\d*\s*(?:mg|ml|mcg|g|iu))?",
    re.IGNORECASE,
)
# Words that legitimately follow a single-letter prefix by coincidence, not
# because it was actually a "Tablet"/"Capsule"/etc. abbreviation.
_MED_STOPWORDS = {"the", "and", "or", "is", "are", "was", "were", "of", "for", "with", "not"}


def _parse_medications(text: str) -> list[dict]:
    seen = set()
    medications = []
    for match in _MED_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name.lower() in _MED_STOPWORDS:
            continue
        dosage = match.group(2).strip() if match.group(2) else None
        key = (name.lower(), dosage)
        if key in seen:
            continue
        seen.add(key)
        entry = {"name": name}
        if dosage:
            entry["dosage"] = dosage
        medications.append(entry)
    return medications


# --- Diagnosis ---------------------------------------------------------------
_DIAGNOSIS_LABEL = r"\b(?:Diagnosis|Dx|Impression)\b"
_DIAGNOSIS_STOP = (
    r"\b(?:BP|Blood\s*Pressure|Blood\s*Sugar|Glucose|Hemoglobin|Haemoglobin|"
    r"Cholesterol|Date|Medications?|Rx|For\s*Medicine)\b"
)
_DIAGNOSIS_PATTERN = re.compile(
    rf"{_DIAGNOSIS_LABEL}\s*[:\-]?\s*(.+?)(?=\s*(?:{_DIAGNOSIS_STOP})|$)",
    re.IGNORECASE,
)


def _parse_diagnosis(text: str) -> str | None:
    match = _DIAGNOSIS_PATTERN.search(text)
    if not match:
        return None
    value = match.group(1).strip(" .,-")
    return value or None


def _mentions_diagnosis_label(text: str) -> bool:
    return bool(re.search(_DIAGNOSIS_LABEL, text, re.IGNORECASE))


def extract_fields(raw_text: str) -> dict:
    """
    Returns a dict of extracted fields plus a per-field `_flags` list noting
    anything that failed a sanity check — the UI should surface these as
    "please verify" rather than silently trusting them.
    """
    text = clean_text(raw_text)
    result: dict = {}
    flags: list[str] = []

    bp_match = _BP_PATTERN.search(text)
    if bp_match:
        split = _BP_SPLIT.search(bp_match.group(1))
        systolic, diastolic = int(split.group(1)), int(split.group(2))
        result["bp_systolic"] = systolic
        result["bp_diastolic"] = diastolic
        if not _in_range("bp_systolic", systolic) or not _in_range("bp_diastolic", diastolic):
            flags.append("bp_out_of_range")

    for key, (pattern, _range, _fuzzy_kw) in _NUMERIC_FIELDS.items():
        m = next(
            (
                c for c in pattern.finditer(text)
                if not _preceded_by_excluded(text, c.start(), key)
                and not _is_range_bound(text, c.start(1))
            ),
            None,
        )
        matched_via_fuzzy = False
        if m:
            raw_value = m.group(1)
        else:
            raw_value = _fuzzy_find(text, key)
            matched_via_fuzzy = raw_value is not None
        if raw_value is None:
            continue
        value = float(raw_value)
        result[key] = value
        if not _in_range(key, value):
            flags.append(f"{key}_out_of_range")
        if matched_via_fuzzy:
            # Extracted via an approximate label match, not an exact one —
            # worth a second look even though it passed its range check.
            flags.append(f"{key}_fuzzy_match")

    for key, pattern in _CATEGORICAL_FIELDS.items():
        m = pattern.search(text)
        if m:
            result[key] = m.group(1).lower()
            # Categorical values can't be range-checked, so always confirm them.
            flags.append(f"{key}_needs_verification")

    date_match = _DATE_PATTERN.search(text)
    if date_match:
        parsed = _parse_date(date_match.group(1))
        if parsed:
            result["date"] = parsed
        else:
            flags.append("date_unparseable")

    medications = _parse_medications(text)
    if medications:
        result["medications"] = medications
        # Drug names can't be range-checked like a number, and OCR errors here
        # are higher-stakes than anywhere else in this file — always surface
        # for a human to confirm, never auto-trust.
        flags.append("medications_need_verification")

    diagnosis = _parse_diagnosis(text)
    if diagnosis:
        result["diagnosis"] = diagnosis
        flags.append("diagnosis_needs_verification")
    elif _mentions_diagnosis_label(text):
        # A label like "Diagnosis:" was seen but nothing usable followed it —
        # surfaced the same way the numeric fields flag a mentioned-but-unparsed value.
        flags.append("unparsed_diagnosis")

    result["_flags"] = flags
    result["_needs_review"] = len(flags) > 0
    return result


if __name__ == "__main__":
    sample = "Patient Report BP, 14O/90 Blood Sugar: 118 CCholesterat 175 Date: 15/O3/2024"
    print(extract_fields(sample))