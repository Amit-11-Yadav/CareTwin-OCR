"""
preprocess.py — Week 2 of the OCR track.

Turns a raw, messy photo of a document (skewed, low-contrast, noisy)
into something an OCR engine can actually read well.

Usage:
    from preprocess import preprocess_image, preprocess_image_light
    cleaned = preprocess_image("raw_prescription.jpg")        # for Tesseract
    cleaned = preprocess_image_light("raw_prescription.jpg")  # for EasyOCR/PaddleOCR
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def correct_contrast(img: np.ndarray) -> np.ndarray:
    """Histogram equalization — helps with poorly lit / low-contrast photos."""
    return cv2.equalizeHist(img)


def denoise(img: np.ndarray) -> np.ndarray:
    """Remove sensor/compression noise without blurring text edges too much."""
    return cv2.fastNlMeansDenoising(img, h=10)


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Detect and correct rotation using the minimum-area bounding box of
    dark (text) pixels. Works well for photos taken at a slight angle —
    the single most common real-world quality issue with phone-camera uploads.
    """
    # Invert so text is white on black, which is what findNonZero expects
    inverted = cv2.bitwise_not(img)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img  # nothing detected, skip deskew rather than crash

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns angles in a range that needs normalizing
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible skew — avoids introducing blur on
    # already-straight images
    if abs(angle) < 0.5:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def binarize(img: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding — converts to clean black/white text.
    Adaptive (not global) thresholding matters here because lighting is
    rarely even across a real photographed document.
    """
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def preprocess_image(path: str, save_debug_path: str | None = None) -> np.ndarray:
    """
    Full pipeline: load -> denoise -> deskew -> contrast -> binarize.

    This is tuned for Tesseract, a classical engine that works on clean
    black/white input. Order matters: denoise before deskew (rotation on
    noisy pixels amplifies noise), deskew before contrast/binarize (so the
    correction has a reasonably clean image to detect angle from).

    Do NOT use this for EasyOCR / PaddleOCR — see preprocess_image_light.
    Deep-learning OCR engines are trained on natural (non-binarized) images
    and do their own internal preprocessing; forcing a hard black/white
    threshold on top throws away grayscale gradient information (e.g. the
    subtle shading inside a shadow) that their networks rely on to tell
    strokes apart, and empirically produces *worse* results, not better.
    """
    img = load_image(path)
    img = denoise(img)
    img = deskew(img)
    img = correct_contrast(img)
    img = binarize(img)

    if save_debug_path:
        cv2.imwrite(save_debug_path, img)

    return img


def preprocess_image_light(path: str, save_debug_path: str | None = None) -> np.ndarray:
    """
    Light pipeline: load -> denoise -> deskew. No contrast equalization or
    binarization.

    Use this for EasyOCR / PaddleOCR. Denoising and rotation correction are
    universally helpful (a tilted or noisy photo is harder for any engine),
    but the hard binarize/contrast steps are Tesseract-specific and measurably
    hurt deep-learning engines — see preprocess_image's docstring.
    """
    img = load_image(path)
    img = denoise(img)
    img = deskew(img)

    if save_debug_path:
        cv2.imwrite(save_debug_path, img)

    return img


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    result = preprocess_image(sys.argv[1], save_debug_path="preprocessed_debug.png")
    print("Saved preprocessed_debug.png — inspect it to sanity-check the pipeline")