from __future__ import annotations

import os

import pytest

from gui_detector_api.detectors.gpa import GPAUltralyticsDetector
from gui_detector_api.detectors.ui_detr import UIDetrDetector
from gui_detector_api.settings import AppSettings, default_models


pytestmark = pytest.mark.slow


def _require_real_model_tests():
    if os.getenv("RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Real model smoke tests are disabled. Set RUN_REAL_MODEL_TESTS=1 to enable them.")


def test_gpa_real_model_load_smoke():
    _require_real_model_tests()
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    detector = GPAUltralyticsDetector("gpa_gui_detector", settings.models["gpa_gui_detector"], settings)
    detector.load()
    assert detector.info.key == "gpa_gui_detector"


def test_ui_detr_real_model_load_smoke():
    _require_real_model_tests()
    settings = AppSettings(active_model="ui_detr_1", models=default_models())
    detector = UIDetrDetector("ui_detr_1", settings.models["ui_detr_1"], settings)
    detector.load()
    assert detector.info.key == "ui_detr_1"
