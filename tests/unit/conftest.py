from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from gui_detector_api.detectors.fake import FakeDetector
from gui_detector_api.detectors.model_registry import ModelRegistry
from gui_detector_api.domain.schemas import BoundingBox, Detection, DetectorBackend
from gui_detector_api.main import create_app
from gui_detector_api.settings import AppSettings, default_models


def make_png_bytes(color: str = "white") -> bytes:
    image = Image.new("RGB", (64, 48), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_settings(
    *,
    root_dir: Path,
    active_model: str = "gpa_gui_detector",
) -> AppSettings:
    models = default_models()
    return AppSettings(
        active_model=active_model,
        models=models,
        model_cache_dir=root_dir / "model-cache-test",
        class_registry_dir=root_dir / "class-registry-test",
    )


def sample_detections() -> list[Detection]:
    return [
        Detection(
            id="det-0001",
            label="button",
            class_id=1,
            confidence=0.92,
            bbox=BoundingBox(x_min=1, y_min=2, x_max=30, y_max=20),
        ),
        Detection(
            id="det-0002",
            label="input",
            class_id=2,
            confidence=0.55,
            bbox=BoundingBox(x_min=10, y_min=22, x_max=50, y_max=42),
        ),
    ]


@pytest.fixture
def png_bytes() -> bytes:
    return make_png_bytes()


@pytest.fixture
def test_settings(tmp_path) -> AppSettings:
    return build_settings(root_dir=tmp_path)


@pytest.fixture
def fake_detector_builder():
    detections = sample_detections()

    def builder(model_key, model_settings, app_settings):
        return FakeDetector(
            model_key,
            model_settings,
            app_settings,
            detections=detections,
        )

    return builder


@pytest.fixture
def ready_app(test_settings, fake_detector_builder):
    registry = ModelRegistry(builders={DetectorBackend.ULTRALYTICS: fake_detector_builder})
    return create_app(settings=test_settings, model_registry=registry)


@pytest.fixture
def not_ready_app(test_settings):
    return create_app(settings=test_settings, model_registry=ModelRegistry(), load_detector_on_startup=False)
