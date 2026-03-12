from __future__ import annotations

import pytest

from gui_detector_api.detectors.base import ModelLoadError
from gui_detector_api.detectors.factory import DetectorFactory
from gui_detector_api.detectors.fake import FakeDetector
from gui_detector_api.detectors.model_registry import ModelRegistry
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.settings import AppSettings, default_models


def test_model_registry_selects_configured_backend():
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    registry = ModelRegistry(
        builders={
            DetectorBackend.ULTRALYTICS: lambda model_key, model_settings, app_settings: FakeDetector(
                model_key, model_settings, app_settings
            ),
        }
    )
    detector = registry.create_active_detector(settings)
    assert detector.model_key == "gpa_gui_detector"


def test_detector_factory_delegates_to_model_registry():
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    registry = ModelRegistry(builders={})
    registry._builders.clear()
    factory = DetectorFactory(registry=registry)
    with pytest.raises(ModelLoadError):
        factory.create(settings)
