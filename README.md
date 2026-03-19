# GUI Detector

FastAPI service for GUI element detection with pluggable detector backends and OCR-powered classification over user-defined classes.

## How it works

1. **Detection** — UI-DETR or GPA finds all UI elements on a screenshot (bounding boxes)
2. **OCR** — EasyOCR reads text from each detected crop (e.g. "Buy Now", "Add to cart")
3. **Classification** — Fuzzy-matches OCR text against your class text exemplars. Falls back to CLIP embedding similarity for non-text elements (icons, images)

## Supported detectors

| Detector | Backend | HuggingFace Repo |
|----------|---------|-----------------|
| GPA GUI Detector | Ultralytics | `Salesforce/GPA-GUI-Detector` |
| UI-DETR-1 | RF-DETR | `racineai/UI-DETR-1` |

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│  Web UI (nginx)     │     │  FastAPI Backend     │
│  localhost:3001     │────▶│  localhost:8000      │
│  web/index.html     │     │  gui_detector_api/   │
└─────────────────────┘     └──────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              UI-DETR/GPA      EasyOCR        CLIP ViT-B/32
              (detection)    (text reading)   (visual fallback)
```

Two Docker services:
- **api** (`Dockerfile.api`) — backend on port **8000**
- **web** (`Dockerfile.frontend`) — nginx frontend on port **3001**, proxies `/v1/*` to API

## API Endpoints

All under `/v1/` prefix.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/healthcheck` | Service metadata |
| GET | `/v1/readiness` | Detector load status |
| GET | `/v1/classes` | List user-defined classes |
| POST | `/v1/classes` | Create class (multipart: name, texts, images) |
| POST | `/v1/classes/batch` | Bulk create classes from JSON |
| PUT | `/v1/classes/{class_id}` | Replace class |
| DELETE | `/v1/classes/{class_id}` | Delete class |
| POST | `/v1/predictions` | Run detection + classification on image |

### Batch class creation

```bash
curl -X POST http://localhost:8000/v1/classes/batch \
  -H "Content-Type: application/json" \
  -d '{
    "classes": [
      {"name": "buy_now", "texts": ["Buy Now"]},
      {"name": "add_to_cart", "texts": ["Add to cart", "Add to basket"]}
    ]
  }'
```

### Prediction response

`POST /v1/predictions` returns:
- `detections` — all detected UI elements with `ocr_text` (what OCR read from each crop)
- `classified_detections` — only detections matching a user-defined class, with `predicted_class`, `similarity_score`, and `match_method` ("ocr" or "visual")
- `classification` — summary (applied, class_count, threshold)

## User-defined classes

Create classes with text exemplars that match button/label text via OCR:

```json
{"name": "buy_now", "texts": ["Buy Now"]}
```

- Text matching is **case-insensitive** and handles word reordering ("now buy" matches "Buy Now")
- Image exemplars can be added for visual similarity (icons, non-text elements)
- Stored on disk: `class-registry/registry.json` + `class-registry/assets/`

## Quick start

```bash
# Install and run locally
poetry install --with dev,models,ocr
poetry run uvicorn gui_detector_api.main:app --reload
# API at http://localhost:8000

# Or run with Docker
docker compose up --build -d
# API at http://localhost:8000, Web UI at http://localhost:3001
```

## Development

```bash
# Install dev dependencies only (enough for tests)
poetry install --with dev

# Run tests
poetry run pytest

# Run a single test
poetry run pytest tests/unit/test_health.py -v

# Run slow model tests (requires models group)
RUN_REAL_MODEL_TESTS=1 poetry run pytest -m slow
```

## Configuration

Defined in `gui_detector_api/settings.py` with pydantic-settings. Env var prefix: `GUI_DETECTOR_`, nested delimiter: `__`.

| Setting | Default | Description |
|---------|---------|-------------|
| `active_model` | `ui_detr_1` | Which detector to load |
| `embedding_model.hf_repo_id` | `openai/clip-vit-base-patch32` | Embedding model for visual fallback |
| `ocr.backend` | `easyocr` | OCR engine (`easyocr`, `tesseract`, `paddleocr`, `none`) |
| `text_match_threshold` | `0.65` | Min fuzzy match score for OCR classification |
| `classification_similarity_threshold` | `0.35` | Min embedding similarity for visual fallback |
| `device` | `auto` | Resolves to `cuda` → `mps` → `cpu` |

Docker persists model weights in the named volume `model-cache`. Health check uses `GET /v1/readiness`.
