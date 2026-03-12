# GUI Detector API

FastAPI service for GUI detector inference with a normalized response schema and two pluggable model backends:

- `Salesforce/GPA-GUI-Detector` through Ultralytics
- `racineai/UI-DETR-1` through RF-DETR

## Endpoints

- `GET /`
- `GET /v1/healthcheck`
- `GET /v1/readiness`
- `POST /v1/predictions`

The API uses the router prefix `/v1`.

`GET /` serves a small built-in UI that uploads one image, calls `POST /v1/predictions`, draws bounding boxes in the browser, and lets you download the annotated canvas as PNG.

`POST /v1/predictions` is JSON-only and returns:

- `model`
- `image`
- `detections`

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

Configuration is defined in [`gui_detector_api/settings.py`](./gui_detector_api/settings.py) with `pydantic-settings`.

The active model is selected with `active_model`, while `models` contains the backend-specific settings for all available model definitions.

- `gpa_gui_detector` uses the Ultralytics runtime for `Salesforce/GPA-GUI-Detector`.
- `ui_detr_1` uses the RF-DETR runtime for `racineai/UI-DETR-1`.
- `device="auto"` resolves to `cuda`, then `mps`, then `cpu`.
- Set `device="mps"` only on Apple Silicon machines where the PyTorch MPS backend is available.
- Docker is CPU-first by default, so `device="auto"` will normally resolve to `cpu` inside the container.
- `docker-compose.yaml` persists downloaded model weights in the named volume `model-cache`.
- Docker Compose now uses the defaults from [`gui_detector_api/settings.py`](./gui_detector_api/settings.py) directly and does not rely on a `.env` override for `active_model`.
- The container health check uses `GET /v1/readiness`, so a model-load failure marks the container as unhealthy instead of silently passing process health.
