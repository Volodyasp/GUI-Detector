from __future__ import annotations


def render_app_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GUI Detector Studio</title>
    <style>
      :root {
        color-scheme: light;
        --ink: #172033;
        --muted: #52607a;
        --paper: #f6f7f4;
        --panel: rgba(255, 255, 255, 0.92);
        --line: rgba(23, 32, 51, 0.12);
        --accent: #0f766e;
        --accent-strong: #115e59;
        --accent-soft: rgba(15, 118, 110, 0.1);
        --warn: #b45309;
        --error: #b42318;
        --shadow: 0 20px 50px rgba(23, 32, 51, 0.12);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 26%),
          radial-gradient(circle at top right, rgba(14, 116, 144, 0.12), transparent 20%),
          linear-gradient(180deg, #f7faf8 0%, #eef4f7 100%);
      }

      main {
        width: min(1280px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 28px 0 40px;
      }

      .hero {
        display: grid;
        gap: 14px;
        margin-bottom: 22px;
      }

      .eyebrow {
        width: fit-content;
        padding: 6px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-strong);
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
      }

      .hero h1 {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        line-height: 0.96;
      }

      .hero p {
        margin: 0;
        max-width: 760px;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
      }

      .layout {
        display: grid;
        gap: 20px;
        grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.95fr);
      }

      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
        overflow: hidden;
        backdrop-filter: blur(16px);
      }

      .panel-head {
        padding: 18px 20px 0;
      }

      .panel-head h2 {
        margin: 0;
        font-size: 1.15rem;
      }

      .panel-head p {
        margin: 6px 0 0;
        color: var(--muted);
        line-height: 1.5;
      }

      .controls {
        display: grid;
        gap: 14px;
        padding: 18px 20px 20px;
      }

      .field {
        display: grid;
        gap: 8px;
      }

      .field label {
        font-size: 0.92rem;
        color: var(--muted);
      }

      input[type="file"] {
        width: 100%;
        padding: 14px;
        border: 1px dashed rgba(15, 118, 110, 0.34);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.84);
        color: var(--ink);
      }

      .button-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }

      button {
        appearance: none;
        border: 0;
        border-radius: 999px;
        padding: 12px 18px;
        font-weight: 700;
        font-size: 0.95rem;
        cursor: pointer;
        transition: transform 160ms ease, opacity 160ms ease, background 160ms ease;
      }

      button:hover:not(:disabled) {
        transform: translateY(-1px);
      }

      button:disabled {
        cursor: not-allowed;
        opacity: 0.55;
      }

      #detect-button {
        background: linear-gradient(135deg, #0f766e 0%, #0b5a7a 100%);
        color: white;
      }

      #download-button {
        background: rgba(23, 32, 51, 0.08);
        color: var(--ink);
      }

      .status {
        min-height: 24px;
        font-size: 0.94rem;
        color: var(--muted);
      }

      .status[data-tone="error"] {
        color: var(--error);
      }

      .status[data-tone="warn"] {
        color: var(--warn);
      }

      .canvas-shell {
        margin: 0 20px 20px;
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid var(--line);
        background:
          linear-gradient(45deg, rgba(15, 118, 110, 0.05) 25%, transparent 25%),
          linear-gradient(-45deg, rgba(15, 118, 110, 0.05) 25%, transparent 25%),
          linear-gradient(45deg, transparent 75%, rgba(15, 118, 110, 0.05) 75%),
          linear-gradient(-45deg, transparent 75%, rgba(15, 118, 110, 0.05) 75%);
        background-size: 28px 28px;
        background-position: 0 0, 0 14px, 14px -14px, -14px 0;
      }

      canvas {
        display: block;
        width: 100%;
        height: auto;
        max-height: 72vh;
      }

      .empty {
        display: grid;
        place-items: center;
        min-height: 320px;
        padding: 28px;
        text-align: center;
        color: var(--muted);
      }

      .meta-grid {
        display: grid;
        gap: 10px;
        padding: 18px 20px 0;
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .meta-card {
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(15, 118, 110, 0.06);
        border: 1px solid rgba(15, 118, 110, 0.12);
      }

      .meta-card span {
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 4px;
      }

      .meta-card strong {
        display: block;
        font-size: 1rem;
      }

      .detections {
        padding: 0 20px 20px;
      }

      table {
        width: 100%;
        border-collapse: collapse;
      }

      th,
      td {
        padding: 12px 0;
        text-align: left;
        border-bottom: 1px solid var(--line);
        font-size: 0.92rem;
      }

      th {
        color: var(--muted);
        font-weight: 600;
      }

      td {
        font-family: "SF Mono", "Menlo", monospace;
      }

      .note {
        padding: 0 20px 20px;
        color: var(--muted);
        font-size: 0.92rem;
      }

      @media (max-width: 980px) {
        .layout {
          grid-template-columns: 1fr;
        }

        .meta-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <div class="eyebrow">GUI Detector Studio</div>
        <h1>Upload a screenshot. Review detections. Download the annotated result.</h1>
        <p>
          This page sends your image to <code>/v1/predictions</code>, receives normalized detections as JSON,
          and draws every bounding box directly in the browser so the API can stay lightweight and JSON-only.
        </p>
      </section>

      <section class="layout">
        <article class="panel">
          <div class="panel-head">
            <h2>Image Workspace</h2>
            <p>Select one PNG or JPEG image, run detection, and save the annotated canvas as a PNG.</p>
          </div>
          <div class="controls">
            <div class="field">
              <label for="image-input">Image file</label>
              <input id="image-input" type="file" accept="image/png,image/jpeg" />
            </div>
            <div class="button-row">
              <button id="detect-button" type="button" disabled>Detect</button>
              <button id="download-button" type="button" disabled>Download PNG</button>
            </div>
            <div id="status-message" class="status" aria-live="polite">Choose an image to begin.</div>
          </div>
          <div id="empty-state" class="empty">
            <div>
              <strong>No image loaded yet.</strong>
              <p>After you choose a file, it will appear here and detections will be drawn on top of it.</p>
            </div>
          </div>
          <div id="canvas-shell" class="canvas-shell" hidden>
            <canvas id="preview-canvas"></canvas>
          </div>
        </article>

        <article class="panel">
          <div class="panel-head">
            <h2>Detection Summary</h2>
            <p>Model metadata and normalized detections from the API response.</p>
          </div>
          <div class="meta-grid">
            <div class="meta-card">
              <span>Active Model</span>
              <strong id="meta-model">—</strong>
            </div>
            <div class="meta-card">
              <span>Backend</span>
              <strong id="meta-backend">—</strong>
            </div>
            <div class="meta-card">
              <span>Detections</span>
              <strong id="meta-count">0</strong>
            </div>
          </div>
          <div class="detections">
            <table>
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Bounding Box</th>
                </tr>
              </thead>
              <tbody id="detections-body">
                <tr>
                  <td colspan="3">No detections yet.</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p id="detections-note" class="note">No detections found.</p>
        </article>
      </section>
    </main>

    <script>
      const fileInput = document.getElementById("image-input");
      const detectButton = document.getElementById("detect-button");
      const downloadButton = document.getElementById("download-button");
      const statusMessage = document.getElementById("status-message");
      const canvas = document.getElementById("preview-canvas");
      const canvasShell = document.getElementById("canvas-shell");
      const emptyState = document.getElementById("empty-state");
      const detectionsBody = document.getElementById("detections-body");
      const detectionsNote = document.getElementById("detections-note");
      const metaModel = document.getElementById("meta-model");
      const metaBackend = document.getElementById("meta-backend");
      const metaCount = document.getElementById("meta-count");
      const context = canvas.getContext("2d");

      const state = {
        file: null,
        image: null,
        objectUrl: null,
        detections: [],
        response: null,
      };

      function setStatus(message, tone = "default") {
        statusMessage.textContent = message;
        if (tone === "default") {
          statusMessage.removeAttribute("data-tone");
        } else {
          statusMessage.dataset.tone = tone;
        }
      }

      function resetSummary() {
        metaModel.textContent = "—";
        metaBackend.textContent = "—";
        metaCount.textContent = "0";
        detectionsBody.innerHTML = '<tr><td colspan="3">No detections yet.</td></tr>';
        detectionsNote.textContent = "No detections found.";
      }

      function clearImageState() {
        state.detections = [];
        state.response = null;
        downloadButton.disabled = true;
        resetSummary();
        if (!state.image) {
          canvas.width = 0;
          canvas.height = 0;
          canvasShell.hidden = true;
          emptyState.hidden = false;
        }
      }

      function updateSummary(payload) {
        metaModel.textContent = payload.model.key;
        metaBackend.textContent = payload.model.backend;
        metaCount.textContent = String(payload.detections.length);

        if (payload.detections.length === 0) {
          detectionsBody.innerHTML = '<tr><td colspan="3">No detections found.</td></tr>';
          detectionsNote.textContent = "No detections found.";
          return;
        }

        detectionsBody.innerHTML = payload.detections.map((detection) => {
          const box = detection.bbox;
          const bbox = [box.x_min, box.y_min, box.x_max, box.y_max]
            .map((value) => Number(value).toFixed(1))
            .join(", ");
          return `
            <tr>
              <td>${escapeHtml(detection.label)}</td>
              <td>${Number(detection.confidence).toFixed(3)}</td>
              <td>${bbox}</td>
            </tr>
          `;
        }).join("");
        detectionsNote.textContent = `${payload.detections.length} detection${payload.detections.length === 1 ? "" : "s"} rendered on the canvas.`;
      }

      function escapeHtml(value) {
        return String(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function drawScene() {
        if (!state.image) {
          return;
        }

        const width = state.image.naturalWidth || state.image.width;
        const height = state.image.naturalHeight || state.image.height;
        canvas.width = width;
        canvas.height = height;
        context.clearRect(0, 0, width, height);
        context.drawImage(state.image, 0, 0, width, height);

        context.lineWidth = Math.max(2, Math.round(width / 320));
        context.strokeStyle = "#0f766e";
        context.font = `${Math.max(14, Math.round(width / 48))}px "SF Mono", Menlo, monospace`;
        context.textBaseline = "top";

        for (const detection of state.detections) {
          const { bbox, label, confidence } = detection;
          const x = Number(bbox.x_min);
          const y = Number(bbox.y_min);
          const boxWidth = Number(bbox.x_max) - x;
          const boxHeight = Number(bbox.y_max) - y;
          const caption = `${label} ${Number(confidence).toFixed(2)}`;
          const textWidth = context.measureText(caption).width;
          const textHeight = Math.max(18, Math.round(width / 42));
          const tagX = x;
          const tagY = Math.max(0, y - textHeight - 8);

          context.fillStyle = "rgba(15, 118, 110, 0.92)";
          context.fillRect(tagX, tagY, textWidth + 16, textHeight + 8);
          context.strokeRect(x, y, boxWidth, boxHeight);
          context.fillStyle = "#ffffff";
          context.fillText(caption, tagX + 8, tagY + 4);
          context.fillStyle = "rgba(15, 118, 110, 0.92)";
        }

        canvasShell.hidden = false;
        emptyState.hidden = true;
      }

      async function loadImage(file) {
        if (state.objectUrl) {
          URL.revokeObjectURL(state.objectUrl);
        }

        state.objectUrl = URL.createObjectURL(file);
        const image = new Image();
        await new Promise((resolve, reject) => {
          image.onload = resolve;
          image.onerror = () => reject(new Error("Could not read the selected image."));
          image.src = state.objectUrl;
        });
        state.image = image;
        drawScene();
      }

      function buildDownloadFilename() {
        if (!state.file) {
          return "annotated.png";
        }

        const name = state.file.name;
        const dotIndex = name.lastIndexOf(".");
        const stem = dotIndex > 0 ? name.slice(0, dotIndex) : name;
        return `${stem}-annotated.png`;
      }

      async function detect() {
        if (!state.file || !state.image) {
          setStatus("Choose an image first.", "warn");
          return;
        }

        detectButton.disabled = true;
        downloadButton.disabled = true;
        setStatus("Running detection...");

        try {
          const formData = new FormData();
          formData.append("image", state.file);

          const response = await fetch("/v1/predictions", {
            method: "POST",
            body: formData,
          });

          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.detail || payload.error || `Request failed with status ${response.status}`);
          }

          state.response = payload;
          state.detections = payload.detections || [];
          updateSummary(payload);
          drawScene();
          downloadButton.disabled = false;
          setStatus(
            state.detections.length === 0
              ? "Detection finished. No detections found."
              : `Detection finished. ${state.detections.length} detection${state.detections.length === 1 ? "" : "s"} found.`
          );
        } catch (error) {
          state.detections = [];
          updateSummary({
            model: { key: "—", backend: "—" },
            detections: [],
          });
          drawScene();
          setStatus(error instanceof Error ? error.message : "Unexpected error during detection.", "error");
        } finally {
          detectButton.disabled = false;
        }
      }

      function downloadCanvas() {
        if (!state.image) {
          setStatus("Load and detect an image before downloading.", "warn");
          return;
        }

        canvas.toBlob((blob) => {
          if (!blob) {
            setStatus("Could not generate a PNG from the current canvas.", "error");
            return;
          }

          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = buildDownloadFilename();
          link.click();
          URL.revokeObjectURL(url);
        }, "image/png");
      }

      fileInput.addEventListener("change", async (event) => {
        const [file] = event.target.files || [];
        state.file = file || null;
        state.detections = [];
        state.response = null;

        if (!file) {
          state.image = null;
          clearImageState();
          detectButton.disabled = true;
          setStatus("Choose an image to begin.");
          return;
        }

        try {
          await loadImage(file);
          clearImageState();
          detectButton.disabled = false;
          setStatus("Image loaded. Click Detect to fetch bounding boxes.");
        } catch (error) {
          state.image = null;
          clearImageState();
          detectButton.disabled = true;
          setStatus(error instanceof Error ? error.message : "Could not read the selected image.", "error");
        }
      });

      detectButton.addEventListener("click", detect);
      downloadButton.addEventListener("click", downloadCanvas);
      resetSummary();
    </script>
  </body>
</html>"""
