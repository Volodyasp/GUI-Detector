from __future__ import annotations

from fastapi.testclient import TestClient

from gui_detector_api.detectors.base import ModelLoadError
from gui_detector_api.detectors.factory import DetectorFactory
from gui_detector_api.detectors.fake import FakeDetector
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.main import create_app
from gui_detector_api.settings import AppSettings, default_models


def test_readiness_reports_load_failure():
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())

    def failing_builder(model_key, model_settings, app_settings):
        return FakeDetector(model_key, model_settings, app_settings, load_error="boom")

    app = create_app(
        settings=settings,
        detector_factory=DetectorFactory(registry={DetectorBackend.ULTRALYTICS: failing_builder}),
    )

    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert "boom" in response.json()["detail"]


def test_readiness_reports_missing_active_model():
    settings = AppSettings(active_model="missing", models=default_models())
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert "not defined" in response.json()["detail"]
