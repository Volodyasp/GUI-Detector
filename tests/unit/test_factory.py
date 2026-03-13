from __future__ import annotations

import pytest

from gui_detector_api.detectors.base import ModelLoadError
from gui_detector_api.detectors.fake import FakeDetector
from gui_detector_api.detectors.model_registry import ModelRegistry
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.settings import AppSettings, default_models


def test_model_registry_selects_configured_backend(tmp_path):
    settings = AppSettings(
        active_model="gpa_gui_detector",
        models=default_models(),
        class_registry_dir=tmp_path / "class-registry",
        model_cache_dir=tmp_path / "model-cache",
    )
    registry = ModelRegistry(
        builders={
            DetectorBackend.ULTRALYTICS: lambda model_key, model_settings, app_settings: FakeDetector(
                model_key, model_settings, app_settings
            ),
        }
    )
    detector = registry.create_active_detector(settings)
    assert detector.model_key == "gpa_gui_detector"


def test_model_registry_rejects_unregistered_backend(tmp_path):
    settings = AppSettings(
        active_model="gpa_gui_detector",
        models=default_models(),
        class_registry_dir=tmp_path / "class-registry",
        model_cache_dir=tmp_path / "model-cache",
    )
    registry = ModelRegistry(builders={})
    registry._builders.clear()
    with pytest.raises(ModelLoadError):
        registry.create_active_detector(settings)
