"""
app.py — Standalone demo web page for the CareTwin OCR pipeline.

Runs completely independently of the team's frontend/backend repos —
just a small Flask app wrapping your existing pipeline.py so you have
something visual to show, without needing to coordinate integration
with anyone else's codebase first.

Setup:
    pip install flask
    python app.py

Then open http://localhost:5000 in your browser.

Assumes this file sits in the SAME folder as your existing pipeline.py,
ocr_engine.py, extract_fields.py, and preprocess.py.
"""

import os
import json
from flask import Flask, request, render_template_string

from pipeline import process_document

app = Flask(__name__)
UPLOAD_FOLDER = "uploads_demo"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CareTwin — OCR Pipeline Demo</title>
<style>
  :root {
    --primary: #004AC6;
    --secondary: #006B5F;
    --bg: #F7F9FB;
    --card: #FFFFFF;
    --text: #191C1E;
    --muted: #5B6472;
    --border: #D8DEE7;
    --good-bg: #E7F6F2;
    --warn-bg: #FFF6E5;
    --warn: #8A5A00;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", Calibri, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
  }
  header {
    background: #0B1120;
    color: white;
    padding: 28px 40px;
  }
  header .kicker {
    color: var(--secondary);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  header h1 {
    margin: 6px 0 0 0;
    font-size: 28px;
  }
  main {
    max-width: 860px;
    margin: 0 auto;
    padding: 32px 24px 60px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 2px 6px rgba(20,30,50,0.06);
  }
  .card h2 {
    margin-top: 0;
    font-size: 16px;
    color: var(--primary);
  }
  input[type=file] {
    display: block;
    margin: 14px 0;
  }
  select, button {
    font-size: 14px;
    padding: 9px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  button {
    background: var(--primary);
    color: white;
    border: none;
    cursor: pointer;
    font-weight: 600;
  }
  button:hover { opacity: 0.92; }
  .metrics {
    display: flex;
    gap: 14px;
    margin: 16px 0;
    flex-wrap: wrap;
  }
  .metric {
    flex: 1;
    min-width: 140px;
    border-radius: 8px;
    padding: 14px 16px;
    border: 1px solid var(--border);
  }
  .metric.good { background: var(--good-bg); border-color: var(--secondary); }
  .metric.warn { background: var(--warn-bg); border-color: var(--warn); }
  .metric .label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--muted);
    text-transform: uppercase;
  }
  .metric .value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
  }
  pre {
    background: #0B1120;
    color: #00E5A0;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
  }
  img.preview {
    max-width: 100%;
    max-height: 380px;
    border-radius: 8px;
    border: 1px solid var(--border);
    display: block;
    margin-top: 12px;
  }
  .flags {
    display: inline-block;
    background: var(--warn-bg);
    color: var(--warn);
    padding: 3px 10px;
    border-radius: 5px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
  }
</style>
</head>
<body>
<header>
  <div class="kicker">CareTwin &middot; OCR / NLP Track</div>
  <h1>Document Pipeline Demo</h1>
</header>
<main>
  <div class="card">
    <h2>Upload a document</h2>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <label for="engine">Engine:&nbsp;</label>
      <select name="engine" id="engine">
        <option value="tesseract" {{ 'selected' if engine == 'tesseract' else '' }}>Tesseract</option>
        <option value="easyocr" {{ 'selected' if engine == 'easyocr' else '' }}>EasyOCR</option>
      </select>
      <br><br>
      <button type="submit">Run OCR Pipeline</button>
    </form>
  </div>

  {% if result %}
  <div class="card">
    <h2>Input Document</h2>
    <img class="preview" src="{{ image_url }}" alt="uploaded document">
  </div>

  <div class="card">
    <h2>Pipeline Result</h2>
    <div class="metrics">
      <div class="metric {{ 'warn' if result.needs_review else 'good' }}">
        <div class="label">Confidence</div>
        <div class="value">{{ result.ocr_confidence }}%</div>
      </div>
      <div class="metric {{ 'warn' if result.needs_review else 'good' }}">
        <div class="label">Needs Review</div>
        <div class="value">{{ result.needs_review }}</div>
      </div>
      <div class="metric {{ 'warn' if result.needs_review else 'good' }}">
        <div class="label">Fields Found</div>
        <div class="value">{{ result.fields|length }}</div>
      </div>
      <div class="metric">
        <div class="label">Engine</div>
        <div class="value" style="font-size:16px;">{{ result.engine }}</div>
      </div>
    </div>

    {% if result.flags %}
      <div>
        {% for f in result.flags %}
          <span class="flags">{{ f }}</span>
        {% endfor %}
      </div>
    {% endif %}

    <h2 style="margin-top:20px;">Extracted Fields</h2>
    <pre>{{ fields_json }}</pre>

    <h2>Raw OCR Text</h2>
    <pre>{{ result.raw_text }}</pre>
  </div>
  {% endif %}

  {% if error %}
  <div class="card" style="border-color:#BA1A1A;">
    <h2 style="color:#BA1A1A;">Error</h2>
    <pre>{{ error }}</pre>
  </div>
  {% endif %}
</main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_url = None
    engine = "tesseract"

    if request.method == "POST":
        engine = request.form.get("engine", "tesseract")
        file = request.files.get("image")
        if file and file.filename:
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            image_url = "/" + save_path.replace("\\", "/")
            try:
                result = process_document(save_path, engine=engine)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

    fields_json = json.dumps(result["fields"], indent=2) if result else "{}"
    return render_template_string(
        PAGE, result=result, error=error, image_url=image_url,
        fields_json=fields_json, engine=engine,
    )


@app.route("/uploads_demo/<path:filename>")
def serve_upload(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
