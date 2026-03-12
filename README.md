# GUI Detector API

FastAPI service for GUI detector inference with a normalized response schema and two pluggable model backends:

- `Salesforce/GPA-GUI-Detector` through Ultralytics
- `racineai/UI-DETR-1` through RF-DETR

## Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /v1/predictions`
- `POST /v1/predictions/preview`

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
