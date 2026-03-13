from __future__ import annotations

from fastapi.testclient import TestClient

from gui_detector_api.detectors.base import ModelLoadError
from gui_detector_api.detectors.fake import FakeDetector
from gui_detector_api.detectors.model_registry import ModelRegistry
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.main import create_app
from gui_detector_api.settings import AppSettings, default_models


def test_readiness_reports_load_failure(tmp_path):
    settings = AppSettings(
        active_model="gpa_gui_detector",
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )

    def failing_builder(model_key, model_settings, app_settings):
        return FakeDetector(model_key, model_settings, app_settings, load_error="boom")

    app = create_app(
        settings=settings,
        model_registry=ModelRegistry(builders={DetectorBackend.ULTRALYTICS: failing_builder}),
    )

    with TestClient(app) as client:
        response = client.get("/v1/readiness")
    assert response.status_code == 503
    assert "boom" in response.json()["detail"]


def test_readiness_reports_missing_active_model(tmp_path):
    settings = AppSettings(
        active_model="missing",
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.get("/v1/readiness")
    assert response.status_code == 503
    assert "not defined" in response.json()["detail"]


def test_readiness_reports_invalid_explicit_device(monkeypatch, tmp_path):
    settings = AppSettings(
        active_model="gpa_gui_detector",
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    settings.models["gpa_gui_detector"].device = "mps"

    class DeviceCheckingDetector(FakeDetector):
        def load(self) -> None:
            self.resolve_device()

    def builder(model_key, model_settings, app_settings):
        return DeviceCheckingDetector(model_key, model_settings, app_settings)

    def fail_device_resolution(*args, **kwargs):
        del args, kwargs
        raise ModelLoadError("Requested device 'mps' is not available in the current environment.")

    monkeypatch.setattr(DeviceCheckingDetector, "resolve_device", fail_device_resolution)

    app = create_app(
        settings=settings,
        model_registry=ModelRegistry(builders={DetectorBackend.ULTRALYTICS: builder}),
    )

    with TestClient(app) as client:
        response = client.get("/v1/readiness")
    assert response.status_code == 503
    assert "Requested device 'mps'" in response.json()["detail"]
