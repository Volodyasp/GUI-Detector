from __future__ import annotations

import pytest

from gui_detector_api.detectors.base import ModelLoadError
from gui_detector_api.detectors.factory import DetectorFactory
from gui_detector_api.detectors.fake import FakeDetector
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.settings import AppSettings, default_models


def test_detector_factory_selects_configured_backend():
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    factory = DetectorFactory(
        registry={
            DetectorBackend.ULTRALYTICS: lambda model_key, model_settings, app_settings: FakeDetector(
                model_key, model_settings, app_settings
            )
        }
    )
    detector = factory.create(settings)
    assert detector.model_key == "gpa_gui_detector"


def test_detector_factory_rejects_unregistered_backend():
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    factory = DetectorFactory(registry={})
    factory._registry.clear()
    with pytest.raises(ModelLoadError):
        factory.create(settings)
