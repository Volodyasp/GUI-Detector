# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI service for GUI element detection with pluggable detector backends (Salesforce GPA-GUI-Detector via Ultralytics, UI-DETR-1 via RF-DETR) and hybrid OCR + visual embedding classification over user-defined classes. Python 3.12+, managed with Poetry.

## Commands

```bash
# Install (core only — enough for tests)
poetry install --with dev

# Install with real model inference
poetry install --with dev,models

# Install with OCR support (EasyOCR + rapidfuzz)
poetry install --with dev,ocr

# Run dev server
poetry run uvicorn gui_detector_api.main:app --reload

# Run all tests
poetry run pytest

# Run a single test file or function
poetry run pytest tests/unit/test_health.py
poetry run pytest tests/unit/test_health.py::test_healthcheck_returns_service_metadata -v

# Run slow integration tests (requires models group + downloaded weights)
RUN_REAL_MODEL_TESTS=1 poetry run pytest -m slow

# Run OCR benchmark on test crops
python scripts/benchmark_ocr.py

# Docker
docker compose up --build
```

## Architecture

**App lifecycle:** `main.py:create_app()` factory → `lifespan.py` async context manager loads the active detector on startup → `runtime.py:RuntimeState` holds all shared state (settings, detector, services).

**Request flow for predictions:** `endpoints.py POST /v1/predictions` → `PredictionService` → `Detector` runs inference → `DetectionClassificationService` classifies detections using OCR-first, visual-embedding-fallback flow via `ClassRegistryService`.

**Classification pipeline (OCR-primary, visual fallback):**
1. Crop each detection from the image
2. Run OCR (`OcrService`) on the crop to extract text
3. If text found: fuzzy-match against class text exemplars (`text_matching.py`)
4. If no text or no match: embed crop with SigLIP (`EmbeddingService`) → KNN against class image exemplars
5. Each `ClassifiedDetection` includes `match_method` ("ocr" or "visual")

**Detector abstraction:** `detectors/base.py:Detector` ABC → concrete implementations (`gpa.py`, `ui_detr.py`). `ModelRegistry` is a factory that maps `DetectorBackend` enum → builder callable. `FakeDetector` used in tests.

**Configuration:** `settings.py:AppSettings` via pydantic-settings. Env var prefix: `GUI_DETECTOR_`. Nested delimiter: `__`. Key settings: `active_model` selects which detector to load, `models` dict holds per-detector config (thresholds, device, HF repo). `ocr` nested settings control OCR backend/language.

**Services layer** (`services/`):
- `prediction.py` — orchestrates detect → classify pipeline
- `classification.py` — OCR-first classification with visual embedding fallback
- `ocr.py` — pluggable OCR backends (EasyOCR, Tesseract, PaddleOCR)
- `text_matching.py` — fuzzy string matching (rapidfuzz or difflib fallback)
- `embeddings.py` — model-agnostic embedding service (SigLIP default, supports any HF model via AutoModel)
- `class_registry.py` — persistent JSON + asset storage on disk for user-defined classes

**Testing:** Tests use `FakeDetector`, `FakeOcrService`, and `create_app()` with injected settings/registry (see `conftest.py`). `httpx.AsyncClient` with `ASGITransport` for endpoint tests. `tmp_path` for isolated disk state.

## API Endpoints

All under `/v1/` prefix. `GET /` serves the built-in HTML UI.
- `GET /v1/healthcheck`, `GET /v1/readiness` — health/status
- `POST /v1/predictions` — multipart image upload, returns detections + classified detections
- `GET|POST|PUT|DELETE /v1/classes[/{class_id}]` — manage user-defined classes with text/image exemplars
