from __future__ import annotations

import base64
import html
import json
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from gui_detector_api.domain.schemas import PredictionResponse


class PreviewRenderer:
    def render(self, prediction: PredictionResponse, image: Image.Image) -> str:
        annotated = self._annotate(image, prediction)
        encoded = self._encode_image(annotated)
        payload_json = html.escape(json.dumps(prediction.model_dump(mode="json"), indent=2))
        rows = "\n".join(self._render_row(detection) for detection in prediction.detections)
        return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GUI Detector Preview</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #1f2937;
        --paper: #f8fafc;
        --panel: #ffffff;
        --line: #d0d9e6;
        --accent: #0f766e;
        --accent-soft: #d1fae5;
      }}
      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 28%),
          linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }}
      .hero {{
        margin-bottom: 24px;
      }}
      .hero h1 {{
        margin: 0 0 8px;
        font-size: 2rem;
      }}
      .hero p {{
        margin: 0;
        color: #475569;
      }}
      .layout {{
        display: grid;
        gap: 20px;
        grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
        overflow: hidden;
      }}
      .panel header {{
        padding: 18px 20px 0;
      }}
      .panel header h2 {{
        margin: 0;
        font-size: 1.1rem;
      }}
      .panel header p {{
        margin: 6px 0 0;
        color: #64748b;
      }}
      img {{
        display: block;
        width: 100%;
        height: auto;
        background: #e2e8f0;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th,
      td {{
        padding: 12px 20px;
        text-align: left;
        border-top: 1px solid var(--line);
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 0.9rem;
      }}
      th {{
        background: #f8fafc;
      }}
      .meta {{
        display: inline-flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
      }}
      .pill {{
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.85rem;
      }}
      pre {{
        margin: 0;
        padding: 16px 20px 20px;
        overflow: auto;
        background: #0f172a;
        color: #d7e5ff;
        font-size: 0.82rem;
      }}
      @media (max-width: 900px) {{
        .layout {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <h1>Prediction Preview</h1>
        <p>Annotated detections for <strong>{html.escape(prediction.model.key)}</strong> rendered from the uploaded image.</p>
        <div class="meta">
          <span class="pill">Backend: {html.escape(prediction.model.backend.value)}</span>
          <span class="pill">Detections: {len(prediction.detections)}</span>
          <span class="pill">Image: {prediction.image.width}x{prediction.image.height}</span>
        </div>
      </section>
      <section class="layout">
        <article class="panel">
          <header>
            <h2>Annotated Image</h2>
            <p>Bounding boxes, labels, and confidence values are overlaid directly onto the source image.</p>
          </header>
          <img alt="Annotated GUI detections" src="data:image/png;base64,{encoded}" />
        </article>
        <article class="panel">
          <header>
            <h2>Detections</h2>
            <p>Normalized prediction payload sorted by descending confidence.</p>
          </header>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Label</th>
                <th>Confidence</th>
                <th>Box</th>
              </tr>
            </thead>
            <tbody>
              {rows}
            </tbody>
          </table>
        </article>
      </section>
      <section class="panel" style="margin-top: 20px;">
        <header>
          <h2>Prediction JSON</h2>
          <p>The exact normalized response returned by the JSON endpoint.</p>
        </header>
        <pre>{payload_json}</pre>
      </section>
    </main>
  </body>
</html>"""

    def _annotate(self, image: Image.Image, prediction: PredictionResponse) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        for detection in prediction.detections:
            box = detection.bbox
            draw.rectangle(
                (box.x_min, box.y_min, box.x_max, box.y_max),
                outline="#0f766e",
                width=3,
            )
            text = f"{detection.label} ({detection.confidence:.2f})"
            anchor = (box.x_min + 4, max(0, box.y_min - 14))
            draw.text(anchor, text, fill="#0f172a", font=font)

        return annotated

    def _encode_image(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _render_row(self, detection) -> str:
        bbox = detection.bbox
        return (
            "<tr>"
            f"<td>{html.escape(detection.id)}</td>"
            f"<td>{html.escape(detection.label)}</td>"
            f"<td>{detection.confidence:.3f}</td>"
            f"<td>{bbox.x_min:.1f}, {bbox.y_min:.1f}, {bbox.x_max:.1f}, {bbox.y_max:.1f}</td>"
            "</tr>"
        )
