from __future__ import annotations

from types import SimpleNamespace

from gui_detector_api.detectors.gpa import GPAUltralyticsDetector
from gui_detector_api.settings import AppSettings, default_models


class FakeYOLOModel:
    names = {0: "button", 1: "input"}

    def __init__(self) -> None:
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        boxes = SimpleNamespace(
            xyxy=[[1, 2, 30, 20], [10, 22, 50, 42]],
            conf=[0.91, 0.52],
            cls=[0, 1],
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


def test_gpa_adapter_normalizes_ultralytics_results(monkeypatch):
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    detector = GPAUltralyticsDetector("gpa_gui_detector", settings.models["gpa_gui_detector"], settings)
    fake_model = FakeYOLOModel()

    monkeypatch.setattr(detector, "resolve_weight_path", lambda: "model.pt")
    monkeypatch.setattr(detector, "resolve_device", lambda: "mps")
    monkeypatch.setattr(detector, "_import_yolo_class", lambda: (lambda _: fake_model))

    detector.load()
    result = detector.predict(image=object())

    assert fake_model.last_kwargs["device"] == "mps"
    assert fake_model.last_kwargs["imgsz"] == settings.models["gpa_gui_detector"].image_size
    assert fake_model.last_kwargs["iou"] == settings.models["gpa_gui_detector"].iou_threshold
    assert result.detections[0].label == "button"
    assert result.detections[1].class_id == 1


def test_gpa_adapter_reports_runtime_dependency_import_errors():
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    detector = GPAUltralyticsDetector("gpa_gui_detector", settings.models["gpa_gui_detector"], settings)

    message = detector._build_runtime_dependency_error(ImportError("libxcb.so.1: cannot open shared object file"))

    assert "libxcb.so.1" in message
    assert "Docker" in message
