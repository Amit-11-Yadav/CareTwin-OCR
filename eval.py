"""
eval.py — objective accuracy measurement for the OCR pipeline.

Runs every image in test_data/images/ through every engine, compares the
extracted fields against test_data/ground_truth.json, and prints a
per-field, per-engine accuracy report.

This exists because OCR confidence scores (mean_confidence) measure how
sure an engine is, not whether it's actually correct — the Tesseract vs.
EasyOCR comparison on the handwritten prescription showed exactly this:
higher confidence did NOT mean more accurate. This script measures the
thing that actually matters: did the pipeline get the right numbers out.

Usage:
    python eval.py
    python eval.py --engines tesseract easyocr
    python eval.py --data-dir test_data
"""

import argparse
import json
from pathlib import Path

from pipeline import process_document

# Numeric fields get a small tolerance since a "correct" OCR read can still
# be off by rounding (e.g. hemoglobin 13.5 vs 13.6). Exact-match fields
# (dates) get none — a wrong date is just wrong.
NUMERIC_TOLERANCE = {
    "bp_systolic": 0,
    "bp_diastolic": 0,
    "blood_sugar": 0,
    "hemoglobin": 0.2,
    "cholesterol": 0,
}


def _field_matches(field: str, expected, actual) -> bool:
    if actual is None:
        return False
    if field in NUMERIC_TOLERANCE:
        try:
            return abs(float(expected) - float(actual)) <= NUMERIC_TOLERANCE[field]
        except (TypeError, ValueError):
            return False
    # Exact match for non-numeric fields (date, diagnosis, medications, etc.)
    return str(expected).strip().lower() == str(actual).strip().lower()


def evaluate(data_dir: str, engines: list[str]) -> None:
    data_path = Path(data_dir)
    ground_truth_path = data_path / "ground_truth.json"
    images_dir = data_path / "images"

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"No ground_truth.json found at {ground_truth_path}")

    ground_truth = json.loads(ground_truth_path.read_text())

    if not ground_truth:
        print("ground_truth.json is empty — add some labeled images first.")
        return

    # results[engine][field] = [correct_count, total_count]
    results = {engine: {} for engine in engines}
    per_image_rows = []

    for image_name, expected_fields in ground_truth.items():
        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"WARNING: {image_path} listed in ground_truth.json but not found — skipping.")
            continue

        for engine in engines:
            try:
                output = process_document(str(image_path), engine=engine)
            except Exception as e:
                print(f"ERROR running {engine} on {image_name}: {e}")
                continue

            actual_fields = output["fields"]

            for field, expected_value in expected_fields.items():
                correct = _field_matches(field, expected_value, actual_fields.get(field))
                bucket = results[engine].setdefault(field, [0, 0])
                bucket[0] += int(correct)
                bucket[1] += 1

                per_image_rows.append({
                    "image": image_name,
                    "engine": engine,
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_fields.get(field),
                    "correct": correct,
                })

    # --- Print per-image detail (useful for debugging specific misses) -----
    print("\n=== Per-field results ===")
    for row in per_image_rows:
        mark = "✓" if row["correct"] else "✗"
        print(f"{mark} [{row['engine']:10s}] {row['image']:20s} {row['field']:15s} "
              f"expected={row['expected']!r:15} actual={row['actual']!r}")

    # --- Print summary table -------------------------------------------------
    print("\n=== Summary: accuracy per engine per field ===")
    all_fields = sorted({f for engine_results in results.values() for f in engine_results})
    header = f"{'field':15s} | " + " | ".join(f"{e:12s}" for e in engines)
    print(header)
    print("-" * len(header))

    for field in all_fields:
        row = [f"{field:15s}"]
        for engine in engines:
            correct, total = results[engine].get(field, (0, 0))
            pct = f"{100 * correct / total:.0f}% ({correct}/{total})" if total else "no data"
            row.append(f"{pct:12s}")
        print(" | ".join(row))

    # --- Overall accuracy per engine -----------------------------------------
    print("\n=== Overall accuracy per engine ===")
    for engine in engines:
        total_correct = sum(c for c, _ in results[engine].values())
        total_count = sum(t for _, t in results[engine].values())
        pct = 100 * total_correct / total_count if total_count else 0
        print(f"{engine:12s}: {pct:.1f}% ({total_correct}/{total_count} fields correct)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR pipeline accuracy against ground truth.")
    parser.add_argument(
        "--engines", nargs="+", default=["tesseract", "easyocr"],
        choices=["tesseract", "easyocr", "paddleocr"],
        help="Which engines to evaluate (default: tesseract easyocr)",
    )
    parser.add_argument(
        "--data-dir", default="test_data",
        help="Folder containing ground_truth.json and images/ (default: test_data)",
    )
    args = parser.parse_args()

    evaluate(args.data_dir, args.engines)
