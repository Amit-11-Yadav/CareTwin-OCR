"""
quality_check.py — the gate that runs BEFORE any OCR engine.

Decides, using cheap image measurements only (no OCR, no LLM, no network),
whether an uploaded document should be:

  * sent through the OCR pipeline  (clean, machine-printed, readable), or
  * stored as-is in the patient's history, unprocessed  (blurry, faint,
    tiny, or handwritten / mixed handwriting).

Why a gate at all: the OCR benchmark showed handwriting and poor photos
come out as confident-looking garbage. In a health record, garbage that
looks like data is worse than no data, so those files are archived
untouched and clearly marked "not machine-read".

All thresholds are module constants so they can be re-tuned against more
real documents. They were calibrated on a small set (clean printed
reports, noisy/rotated printed reports, one real handwritten prescription,
and synthetically blurred / faint / dark / low-res copies), so treat them
as a sensible starting point, not a finished classifier.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np

# --- Thresholds (tune against real uploads) --------------------------------
MIN_WIDTH_PX = 400          # narrower than this is too small to read reliably
MIN_BLUR_SCORE = 50.0       # variance of Laplacian at normalised width; clean ~150-550, blurred ~0-20
MIN_CONTRAST = 60.0         # gap between background and ink brightness (0-255)
MAX_HEIGHT_CV = 0.65        # spread of character heights; simple printed ~0.1-0.25, handwriting ~0.7
# NOTE: this value is a rough estimate, not a settled constant. It was raised
# from 0.45 after a REAL multi-section lab report (large header + small table
# text + colored status words) scored 0.61 and was wrongly classified as
# handwritten — mixing font sizes by section is normal in real printed
# reports and drives this metric up even with zero handwriting present. The
# only real handwriting sample tested scored 0.70, leaving just a 0.09 gap
# between one real printed counter-example and one real handwriting sample.
# Treat this threshold as unverified until tested against several more real
# documents of both kinds — it is very possible it still needs adjusting.

_NORM_WIDTH = 1000          # metrics are measured at this width so photo size doesn't skew them


@dataclass
class QualityReport:
    passed: bool                     # True -> safe to send to OCR
    doc_type: str                    # "printed" | "handwritten_or_mixed" | "unusable"
    reasons: list = field(default_factory=list)   # empty when passed
    metrics: dict = field(default_factory=dict)


def _normalise_width(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = _NORM_WIDTH / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (_NORM_WIDTH, max(1, int(h * scale))), interpolation=interp)


def _ink_on_light(img: np.ndarray) -> np.ndarray:
    """Dark-theme screenshots (light text on dark) are inverted so every check sees dark ink on light paper."""
    return cv2.bitwise_not(img) if img.mean() < 127 else img


def blur_score(img: np.ndarray) -> float:
    """Variance of the Laplacian: high when edges are sharp, near zero when blurred."""
    return float(cv2.Laplacian(_normalise_width(img), cv2.CV_64F).var())


def contrast_score(img: np.ndarray) -> float:
    """
    Brightness gap between background and ink, using Otsu's split so it works
    whether ink covers 2% of the page (a typed report) or 30% (dense text).
    """
    thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = img[img <= thresh]
    light = img[img > thresh]
    if dark.size == 0 or light.size == 0:
        return 0.0
    return float(abs(light.mean() - dark.mean()))


def character_height_cv(img: np.ndarray) -> float:
    """
    Coefficient of variation of connected-component heights.

    Printed text reuses a few fixed glyph heights, so heights cluster tightly
    (low value). Handwriting varies continuously in size and slant (high
    value). Returns 0.0 if too few characters are found to judge.
    """
    g = _normalise_width(_ink_on_light(img))
    binary = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    page_h, page_w = g.shape
    heights = [
        stats[i][cv2.CC_STAT_HEIGHT]
        for i in range(1, n)
        if stats[i][cv2.CC_STAT_AREA] >= 15
        and stats[i][cv2.CC_STAT_HEIGHT] >= 6
        and stats[i][cv2.CC_STAT_HEIGHT] < page_h * 0.3
        and stats[i][cv2.CC_STAT_WIDTH] < page_w * 0.3
    ]
    if len(heights) < 10:
        return 0.0
    heights = np.array(heights, dtype=float)
    return float(heights.std() / heights.mean())


def assess_quality(img: np.ndarray) -> QualityReport:
    """
    img: grayscale image array (as loaded by cv2.imread(..., IMREAD_GRAYSCALE)).
    Returns a QualityReport; report.passed is True only if every check passes.
    """
    if img is None or img.size == 0:
        raise ValueError("assess_quality received an empty image")

    metrics = {
        "width_px": int(img.shape[1]),
        "blur_score": round(blur_score(img), 1),
        "contrast": round(contrast_score(img), 1),
        "height_cv": round(character_height_cv(img), 2),
    }

    reasons = []
    if metrics["width_px"] < MIN_WIDTH_PX:
        reasons.append("low_resolution")
    if metrics["blur_score"] < MIN_BLUR_SCORE:
        reasons.append("too_blurry")
    if metrics["contrast"] < MIN_CONTRAST:
        reasons.append("low_contrast")
    handwritten = metrics["height_cv"] > MAX_HEIGHT_CV
    if handwritten:
        reasons.append("handwritten_or_mixed")

    if handwritten:
        doc_type = "handwritten_or_mixed"
    elif reasons:
        doc_type = "unusable"
    else:
        doc_type = "printed"

    return QualityReport(passed=not reasons, doc_type=doc_type, reasons=reasons, metrics=metrics)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python quality_check.py <image_path>")
        sys.exit(1)
    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image at {sys.argv[1]}")
        sys.exit(1)
    report = assess_quality(image)
    print(f"route:   {'OCR pipeline' if report.passed else 'store only (not machine-read)'}")
    print(f"type:    {report.doc_type}")
    print(f"reasons: {report.reasons or '-'}")
    print(f"metrics: {report.metrics}")