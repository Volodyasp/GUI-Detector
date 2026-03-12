from __future__ import annotations

from types import SimpleNamespace

from gui_detector_api.detectors.gpa import GPAUltralyticsDetector
from gui_detector_api.settings import AppSettings, default_models


class FakeYOLOModel:
    names = {0: "button", 1: "input"}

    def predict(self, **kwargs):
        del kwargs
        boxes = SimpleNamespace(
            xyxy=[[1, 2, 30, 20], [10, 22, 50, 42]],
            conf=[0.91, 0.52],
            cls=[0, 1],
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


def test_gpa_adapter_normalizes_ultralytics_results(monkeypatch):
    settings = AppSettings(active_model="gpa_gui_detector", models=default_models())
    detector = GPAUltralyticsDetector("gpa_gui_detector", settings.models["gpa_gui_detector"], settings)

    monkeypatch.setattr(detector, "resolve_weight_path", lambda: "model.pt")
    monkeypatch.setattr(detector, "_import_yolo_class", lambda: (lambda _: FakeYOLOModel()))

    detector.load()
    result = detector.predict(image=object())

    assert result.detections[0].label == "button"
    assert result.detections[1].class_id == 1
