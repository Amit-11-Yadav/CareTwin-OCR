"""
ocr_engine.py — Week 1 (engine bake-off) + Week 3 (wiring the winner in).

One consistent interface (`run_ocr`) regardless of which engine backs it,
so swapping engines later doesn't touch any other file in the pipeline.

Three engines are wired up: Tesseract, EasyOCR, and PaddleOCR. EasyOCR and
PaddleOCR are imported lazily (inside their run_ocr_<engine> function, not
at module load) and their reader objects are built once and cached in a
module-level global — both are deep-learning models with real load time
and memory footprint, so you don't want that cost paid just for importing
this file, or paid again on every single call.

Install (not needed for Tesseract, which is already required):
    pip install easyocr --break-system-packages
    pip install paddlepaddle paddleocr --break-system-packages
"""

from dataclasses import dataclass
import numpy as np
import cv2
import pytesseract


@dataclass
class OCRResult:
    text: str
    mean_confidence: float  # 0-100; use this to drive the "please verify" badge
    engine: str


def run_ocr_tesseract(img: np.ndarray) -> OCRResult:
    """
    Runs Tesseract and also pulls per-word confidence scores, so we can
    compute one overall confidence number for the "Extracted" vs.
    "Please verify" badge on the document card.
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    confidences = []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        conf = float(conf)
        if word and conf >= 0:  # -1 confidence means "not a real word", skip it
            words.append(word)
            confidences.append(conf)

    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(text=text, mean_confidence=mean_conf, engine="tesseract")


# --- EasyOCR ----------------------------------------------------------------
#
# Cached at module level: constructing easyocr.Reader loads the detection +
# recognition models from disk (or downloads them on first run), which takes
# real time. Build it once, reuse it across every call in the process.
_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # deferred: avoid paying import cost if unused
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def run_ocr_easyocr(img: np.ndarray) -> OCRResult:
    """
    EasyOCR does its own detection + recognition (no separate binarize step
    needed — it handles color/grayscale input directly). readtext returns
    (bounding_box, text, confidence) tuples; confidence is already 0-1.
    """
    reader = _get_easyocr_reader()
    results = reader.readtext(img)

    words = []
    confidences = []
    for (_, word, conf) in results:
        word = word.strip()
        if word:
            words.append(word)
            # EasyOCR's conf comes back as a numpy float, not a plain Python
            # float. Cast now so numpy types never leak downstream — they
            # silently produce numpy.bool_ in later comparisons, which
            # json.dumps() cannot serialize.
            confidences.append(float(conf) * 100)

    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(text=text, mean_confidence=mean_conf, engine="easyocr")


# --- PaddleOCR ----------------------------------------------------------------
#
# Same caching rationale as EasyOCR above.
#
# NOTE: PaddleOCR's API changed substantially in its 3.x release — this is
# written against that current API (paddleocr>=3.0). If you're on an older
# 2.x install, use_angle_cls/show_log and reader.ocr(img, cls=True) were the
# old equivalents; upgrading (pip install -U paddleocr) is simpler than
# maintaining both.
_paddle_reader = None


def _get_paddle_reader():
    global _paddle_reader
    if _paddle_reader is None:
        from paddleocr import PaddleOCR  # deferred: avoid paying import cost if unused
        # use_textline_orientation replaces the old use_angle_cls — handles
        # upside-down/rotated text, which Tesseract doesn't do on its own.
        # show_log no longer exists as an argument in 3.x.
        _paddle_reader = PaddleOCR(lang="en", use_textline_orientation=True)
    return _paddle_reader


def run_ocr_paddle(img: np.ndarray) -> OCRResult:
    """
    PaddleOCR expects a 3-channel image, but pipeline.py loads everything as
    grayscale (cv2.IMREAD_GRAYSCALE) to keep Tesseract/EasyOCR/preprocess.py
    simple — so convert here rather than changing what every other engine
    receives.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    reader = _get_paddle_reader()
    result = reader.predict(img)  # 3.x: .predict() replaces the old .ocr()

    words = []
    confidences = []
    if result:
        page = result[0]
        rec_texts = page["rec_texts"] if "rec_texts" in page else []
        rec_scores = page["rec_scores"] if "rec_scores" in page else []
        for word, conf in zip(rec_texts, rec_scores):
            word = word.strip()
            if word:
                words.append(word)
                confidences.append(float(conf) * 100)  # see easyocr note on numpy types

    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(text=text, mean_confidence=mean_conf, engine="paddleocr")


def run_ocr(img: np.ndarray, engine: str = "tesseract") -> OCRResult:
    """
    Single entry point the rest of the pipeline calls. extract_fields.py and
    pipeline.py are agnostic to which engine produced the text — swap the
    `engine` argument and nothing downstream needs to change.
    """
    if engine == "tesseract":
        return run_ocr_tesseract(img)
    if engine == "easyocr":
        return run_ocr_easyocr(img)
    if engine == "paddleocr":
        return run_ocr_paddle(img)
    raise ValueError(
        f"Unknown engine '{engine}'. Expected one of: tesseract, easyocr, paddleocr."
    )