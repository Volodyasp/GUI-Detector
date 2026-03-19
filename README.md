# GUI Detector API

FastAPI service for GUI detector inference with a normalized response schema, two pluggable detector backends, and optional OCR-primary post-classification with embedding fallback over user-defined classes.

## Supported detectors

- `Salesforce/GPA-GUI-Detector` through Ultralytics
- `racineai/UI-DETR-1` through RF-DETR

Post-classification uses an OCR-primary flow: detected crops are first matched against text exemplars via fuzzy OCR matching, then unmatched crops fall back to embedding-based similarity using `openai/clip-vit-base-patch32`. Embeddings are used only for cosine-similarity classification, not for object detection.

## Endpoints

- `GET /`
- `GET /v1/healthcheck`
- `GET /v1/readiness`
- `GET /v1/classes`
- `POST /v1/classes`
- `PUT /v1/classes/{class_id}`
- `DELETE /v1/classes/{class_id}`
- `POST /v1/predictions`

The API uses the router prefix `/v1`.

`GET /` serves a small built-in UI that uploads one image, calls `POST /v1/predictions`, draws the final accepted detections in the browser, and lets you download the annotated canvas as PNG.

`POST /v1/predictions` is JSON-only and returns:

- `model`
- `image`
- `detections`
- `classified_detections`
- `classification`

`detections` is the raw detector output. `classified_detections` contains only the detections that matched a user-defined class through OCR text matching or embedding cosine similarity with KNN scoring. If no classes are defined, `classification.applied` is `false` and `classified_detections` is empty.

Multipart request fields for `POST /v1/predictions`:

- `image`: required target image

OWLv2 reference inputs are no longer part of the API. Requests that still send `query_texts` or `query_image` receive `400`.

## User-defined classes

Classes are managed through backend API endpoints and stored on disk:

- registry JSON: `class-registry/registry.json`
- saved exemplar images: `class-registry/assets/<class_id>/...`

Each class can contain:

- `name`
- `texts[]`
- `images[]`

At least one text or image exemplar is required. Texts and images are embedded with `openai/clip-vit-base-patch32`, stored as normalized vectors, and reused for prediction-time post-classification.

## Local Development

Install the core app and dev dependencies:

```bash
poetry install --with dev
```

Install optional model runtimes when you want real inference locally:

```bash
poetry install --with dev,models
```

Run the API:

```bash
poetry run uvicorn gui_detector_api.main:app --reload
```

Open the built-in UI:

```bash
http://localhost:8000/
```

Run with Docker Compose:

```bash
docker compose up --build
```

Rebuild the image after Dockerfile changes:

```bash
docker compose build --no-cache
docker compose up -d
```

Run tests:

```bash
poetry run pytest
```

Run the optional slow model smoke tests:

```bash
RUN_REAL_MODEL_TESTS=1 poetry run pytest -m slow
```

## Configuration

Configuration is defined in [`/Users/vladimir/Projects/GUI-Detector/gui_detector_api/settings.py`](/Users/vladimir/Projects/GUI-Detector/gui_detector_api/settings.py) with `pydantic-settings`.

- `active_model` selects the detector runtime.
- `models` contains detector-specific settings for GPA and UI-DETR.
- `embedding_model` configures CLIP embeddings through `openai/clip-vit-base-patch32`.
- `classification_knn_k` controls how many nearest exemplar embeddings are considered per crop.
- `classification_similarity_threshold` is the acceptance threshold for a classified detection.
- `class_registry_dir` controls where the persistent class registry and assets are stored.
- `device="auto"` resolves to `cuda`, then `mps`, then `cpu`.
- Set `device="mps"` only on Apple Silicon machines where the PyTorch MPS backend is available.
- On macOS, `device="auto"` can resolve to `mps` only when you run the app directly on the host with a native PyTorch install.
- Docker Desktop on Mac runs Linux containers, and the PyTorch MPS backend is macOS-only, so `device="auto"` will normally resolve to `cpu` inside the container.
- `docker compose` persists downloaded model weights in the named volume `model-cache`.
- The container health check uses `GET /v1/readiness`, so a detector load failure marks the container as unhealthy instead of silently passing process health.
