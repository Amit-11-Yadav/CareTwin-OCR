"""
app.py — Standalone demo web page for the CareTwin OCR pipeline.

Designed as a diagnostic instrument panel rather than a generic dashboard —
the confidence score is the one reading this whole tool exists to produce,
so it's rendered as a calibrated gauge, not a stat card.

Setup:
    pip install flask
    python app.py

Then open http://localhost:5000 in your browser.

Assumes this file sits in the SAME folder as your existing pipeline.py,
ocr_engine.py, extract_fields.py, and preprocess.py.
"""

import os
import math
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import process_document
import quality_check as qc

app = Flask(__name__)
UPLOAD_FOLDER = "uploads_demo"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GAUGE_RADIUS = 70
GAUGE_CIRCUMFERENCE = 2 * math.pi * GAUGE_RADIUS

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Console — CareTwin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #0A0E14;
    --panel: #10151D;
    --panel-raised: #141B25;
    --line: #232B36;
    --line-soft: #1A212B;
    --blue: #2E7DFF;
    --teal: #22C3A6;
    --amber: #F5A623;
    --red: #FF5C5C;
    --text: #E8ECF2;
    --muted: #7C8798;
    --faint: #4A5568;
  }
  * { box-sizing: border-box; }
  @media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
  }
  body {
    margin: 0;
    background: var(--ink);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
  }
  .mono { font-family: 'IBM Plex Mono', monospace; }

  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    min-height: 100vh;
  }
  @media (max-width: 820px) {
    .layout { grid-template-columns: 1fr; }
  }

  /* ---- Control rail ---- */
  aside {
    background: var(--panel);
    border-right: 1px solid var(--line);
    padding: 28px 24px;
  }
  .brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.5px;
    color: var(--blue);
    margin-bottom: 2px;
  }
  .brand-sub {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 28px;
    line-height: 1.3;
  }
  .field-group { margin-bottom: 18px; }
  .field-label {
    display: block;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 7px;
    font-family: 'IBM Plex Mono', monospace;
  }
  input[type=file] {
    width: 100%;
    color: var(--muted);
    font-size: 12.5px;
    background: var(--panel-raised);
    border: 1px solid var(--line);
    padding: 10px;
    border-radius: 3px;
  }
  select {
    width: 100%;
    background: var(--panel-raised);
    border: 1px solid var(--line);
    color: var(--text);
    padding: 10px 12px;
    border-radius: 3px;
    font-size: 13.5px;
    font-family: 'IBM Plex Sans', sans-serif;
  }
  select:focus, input:focus, button:focus {
    outline: 2px solid var(--blue);
    outline-offset: 1px;
  }
  button.run {
    width: 100%;
    background: var(--blue);
    color: #05070A;
    border: none;
    padding: 13px;
    border-radius: 3px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    margin-top: 6px;
  }
  button.run:hover { background: #4F92FF; }

  .rail-note {
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid var(--line);
    font-size: 12px;
    color: var(--faint);
    line-height: 1.6;
  }

  /* ---- Main readout ---- */
  main { padding: 32px 40px 60px; max-width: 760px; }
  .panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--muted);
    letter-spacing: 0.5px;
    margin: 0 0 10px 0;
  }
  section.block { margin-bottom: 34px; }

  .preview-frame {
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 6px;
    background: var(--panel);
  }
  .preview-frame img {
    display: block;
    width: 100%;
    max-height: 340px;
    object-fit: contain;
    border-radius: 2px;
  }

  /* ---- Gauge (the hero) ---- */
  .gauge-row {
    display: flex;
    align-items: center;
    gap: 32px;
    padding: 8px 0 4px;
    animation: reveal 0.5s ease-out;
  }
  @keyframes reveal {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .gauge-wrap { position: relative; width: 168px; height: 168px; flex-shrink: 0; }
  .gauge-wrap svg { transform: rotate(-90deg); }
  .gauge-track { fill: none; stroke: var(--line); stroke-width: 12; }
  .gauge-value { fill: none; stroke-width: 12; stroke-linecap: round; transition: stroke-dashoffset 0.6s ease; }
  .gauge-center {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }
  .gauge-number { font-family: 'IBM Plex Mono', monospace; font-size: 30px; font-weight: 600; }
  .gauge-unit { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--muted); margin-top: 2px; }

  .status-col { flex: 1; }
  .status-verdict { font-size: 17px; font-weight: 600; margin-bottom: 6px; }
  .status-detail { color: var(--muted); font-size: 13px; line-height: 1.5; }

  .readout-strip {
    display: flex;
    gap: 0;
    margin-top: 22px;
    border: 1px solid var(--line);
    border-radius: 4px;
    overflow: hidden;
  }
  .readout-cell {
    flex: 1;
    padding: 12px 16px;
    border-right: 1px solid var(--line);
  }
  .readout-cell:last-child { border-right: none; }
  .readout-cell .label { font-size: 10.5px; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }
  .readout-cell .val { font-family: 'IBM Plex Mono', monospace; font-size: 14px; margin-top: 3px; }

  .flag-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--amber);
    border: 1px solid var(--amber);
    background: rgba(245,166,35,0.08);
    padding: 3px 9px;
    border-radius: 3px;
    margin: 4px 6px 0 0;
  }

  table.fields {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
  }
  table.fields td {
    padding: 9px 4px;
    border-bottom: 1px solid var(--line-soft);
  }
  table.fields td:first-child { color: var(--muted); width: 45%; }
  table.fields td:last-child { font-family: 'IBM Plex Mono', monospace; }
  table.fields tr:last-child td { border-bottom: none; }

  .raw-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    line-height: 1.7;
    color: #A8B3C2;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 16px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .stored-panel {
    border: 1px solid var(--blue);
    background: rgba(46,125,255,0.06);
    border-radius: 4px;
    padding: 20px 22px;
  }
  .stored-panel .verdict { font-size: 17px; font-weight: 600; color: var(--blue); margin-bottom: 6px; }
  .stored-panel .detail { color: var(--muted); font-size: 13px; line-height: 1.6; }
  table.metrics { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px; }
  table.metrics th { text-align: left; font-weight: 500; color: var(--muted); font-size: 10.5px;
    font-family: 'IBM Plex Mono', monospace; padding: 6px 4px; border-bottom: 1px solid var(--line); }
  table.metrics td { padding: 8px 4px; border-bottom: 1px solid var(--line-soft); font-family: 'IBM Plex Mono', monospace; }
  table.metrics td:first-child { color: var(--muted); font-family: 'IBM Plex Sans', sans-serif; }
  .ok { color: var(--teal); } .bad { color: var(--amber); }

  .empty-state {
    color: var(--faint);
    font-size: 13.5px;
    padding: 40px 0;
  }

  .error-panel {
    border: 1px solid var(--red);
    background: rgba(255,92,92,0.06);
    border-radius: 4px;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: #FF8A8A;
  }
</style>
</head>
<body>
<div class="layout">

  <aside>
    <div class="brand">CARETWIN</div>
    <div class="brand-sub">OCR Console</div>

    <form method="POST" enctype="multipart/form-data">
      <div class="field-group">
        <label class="field-label" for="image">DOCUMENT</label>
        <input type="file" id="image" name="image" accept="image/*" required>
      </div>
      <div class="field-group">
        <label class="field-label" for="engine">ENGINE</label>
        <select name="engine" id="engine">
          <option value="tesseract" {{ 'selected' if engine == 'tesseract' else '' }}>Tesseract</option>
          <option value="easyocr" {{ 'selected' if engine == 'easyocr' else '' }}>EasyOCR</option>
        </select>
      </div>
      <button class="run" type="submit">Run pipeline</button>
    </form>

    <div class="rail-note">
      Blurry, low-contrast, or handwritten documents are stored as-is without OCR. Clean printed documents are read, and anything with low confidence or a missing value is flagged for manual review.
    </div>
  </aside>

  <main>
    {% if result %}
      <section class="block">
        <p class="panel-title">INPUT</p>
        <div class="preview-frame">
          <img src="{{ image_url }}" alt="uploaded document">
        </div>
      </section>

      {% if result.status == 'stored_only' %}
      <section class="block">
        <p class="panel-title">STORAGE</p>
        <div class="stored-panel">
          <div class="verdict">Stored in patient history, not machine-read</div>
          <div class="detail">{{ stored_message }} The original image is kept as part of the patient's record; no text was extracted from it, so its contents are not included in any automatic summary.</div>
        </div>
      </section>

      <section class="block">
        <p class="panel-title">QUALITY CHECK</p>
        <table class="metrics">
          <tr><th>MEASURE</th><th>VALUE</th><th>REQUIRED</th><th>RESULT</th></tr>
          {% for m in quality_rows %}
            <tr><td>{{ m.label }}</td><td>{{ m.value }}</td><td>{{ m.rule }}</td>
                <td class="{{ 'ok' if m.ok else 'bad' }}">{{ 'pass' if m.ok else 'fail' }}</td></tr>
          {% endfor %}
        </table>
      </section>
      {% else %}
      <section class="block">
        <p class="panel-title">CONFIDENCE</p>
        <div class="gauge-row">
          <div class="gauge-wrap">
            <svg width="168" height="168" viewBox="0 0 168 168">
              <circle class="gauge-track" cx="84" cy="84" r="{{ gauge_radius }}"></circle>
              <circle class="gauge-value" cx="84" cy="84" r="{{ gauge_radius }}"
                stroke="{{ status_color }}"
                stroke-dasharray="{{ gauge_circumference }}"
                stroke-dashoffset="{{ gauge_offset }}"></circle>
            </svg>
            <div class="gauge-center">
              <div class="gauge-number">{{ result.ocr_confidence }}</div>
              <div class="gauge-unit">PERCENT</div>
            </div>
          </div>
          <div class="status-col">
            <div class="status-verdict" style="color: {{ status_color }};">
              {{ 'Flagged for review' if result.needs_review else 'Reads as trustworthy' }}
            </div>
            <div class="status-detail">
              {% if result.needs_review %}
                {{ review_message }}
              {% else %}
                Confidence and all extracted values passed their checks. No manual review needed.
              {% endif %}
            </div>
          </div>
        </div>

        <div class="readout-strip">
          <div class="readout-cell">
            <div class="label">ENGINE</div>
            <div class="val">{{ result.engine }}</div>
          </div>
          <div class="readout-cell">
            <div class="label">FIELDS FOUND</div>
            <div class="val">{{ result.fields|length }}</div>
          </div>
          <div class="readout-cell">
            <div class="label">FLAGS</div>
            <div class="val">{{ result.flags|length or '—' }}</div>
          </div>
        </div>

        {% if result.flags %}
          <div style="margin-top:10px;">
            {% for f in result.flags %}
              <span class="flag-tag">{{ f }}</span>
            {% endfor %}
          </div>
        {% endif %}
      </section>

      <section class="block">
        <p class="panel-title">EXTRACTED FIELDS</p>
        {% if result.fields %}
          <table class="fields">
            {% for k, v in result.fields.items() %}
              <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
            {% endfor %}
          </table>
        {% else %}
          <div class="empty-state">No fields matched. Check the raw text below to see what the engine actually read.</div>
        {% endif %}
      </section>

      <section class="block">
        <p class="panel-title">RAW OCR TEXT</p>
        <div class="raw-text">{{ result.raw_text or '(empty)' }}</div>
      </section>
      {% endif %}

    {% elif error %}
      <section class="block">
        <p class="panel-title">ERROR</p>
        <div class="error-panel">{{ error }}</div>
      </section>
    {% else %}
      <div class="empty-state">Upload a document and run the pipeline to see a reading here.</div>
    {% endif %}
  </main>

</div>
</body>
</html>
"""


STORED_MESSAGES = {
    "handwritten_or_mixed": "Handwriting was detected, and OCR is not reliable on handwriting.",
    "too_blurry": "The image is too blurry to read reliably.",
    "low_contrast": "The text is too faint or too dark against the background.",
    "low_resolution": "The image resolution is too low to read reliably.",
}


def stored_message(flags):
    parts = [STORED_MESSAGES[f] for f in flags if f in STORED_MESSAGES]
    return " ".join(parts) or "The image did not meet the quality requirements for OCR."


def review_message(flags):
    """Say specifically WHY a processed result needs review, instead of one generic sentence."""
    missing = [f.replace("unparsed_", "").replace("_", " ") for f in flags if f.startswith("unparsed_")]
    other = [f for f in flags if not f.startswith("unparsed_")]
    msgs = []
    if missing:
        msgs.append("The text mentions " + ", ".join(missing) + " but no value could be read for it, so this extraction is incomplete.")
    if other:
        msgs.append("An extracted value fell outside its expected range.")
    if not msgs:
        msgs.append("OCR confidence is below the review threshold.")
    return " ".join(msgs) + " Verify before using this reading."


def quality_rows(q):
    return [
        {"label": "Width (px)", "value": q["width_px"], "rule": f">= {qc.MIN_WIDTH_PX}", "ok": q["width_px"] >= qc.MIN_WIDTH_PX},
        {"label": "Sharpness", "value": q["blur_score"], "rule": f">= {qc.MIN_BLUR_SCORE:g}", "ok": q["blur_score"] >= qc.MIN_BLUR_SCORE},
        {"label": "Contrast", "value": q["contrast"], "rule": f">= {qc.MIN_CONTRAST:g}", "ok": q["contrast"] >= qc.MIN_CONTRAST},
        {"label": "Print regularity (lower = printed)", "value": q["height_cv"], "rule": f"<= {qc.MAX_HEIGHT_CV}", "ok": q["height_cv"] <= qc.MAX_HEIGHT_CV},
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_url = None
    engine = "tesseract"
    gauge_offset = GAUGE_CIRCUMFERENCE
    status_color = "#22C3A6"
    stored_msg = ""
    review_msg = ""
    q_rows = []

    if request.method == "POST":
        engine = request.form.get("engine", "tesseract")
        file = request.files.get("image")
        if file and file.filename:
            save_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename) or "upload")
            file.save(save_path)
            image_url = "/" + save_path.replace("\\", "/")
            try:
                result = process_document(save_path, engine=engine)
                pct = max(0, min(100, result["ocr_confidence"]))
                gauge_offset = GAUGE_CIRCUMFERENCE * (1 - pct / 100)
                status_color = "#F5A623" if result["needs_review"] else "#22C3A6"
                if result["status"] == "stored_only":
                    stored_msg = stored_message(result["flags"])
                    q_rows = quality_rows(result["quality"])
                elif result["needs_review"]:
                    review_msg = review_message(result["flags"])
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

    return render_template_string(
        PAGE, result=result, error=error, image_url=image_url, engine=engine,
        gauge_radius=GAUGE_RADIUS, gauge_circumference=round(GAUGE_CIRCUMFERENCE, 2),
        gauge_offset=round(gauge_offset, 2), status_color=status_color,
        stored_message=stored_msg, review_message=review_msg, quality_rows=q_rows,
    )


@app.route("/uploads_demo/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)