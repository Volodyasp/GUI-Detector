from __future__ import annotations

from types import SimpleNamespace

from gui_detector_api.detectors.ui_detr import UIDetrDetector
from gui_detector_api.settings import AppSettings, default_models


class FakeTorch:
    @staticmethod
    def load(*args, **kwargs):
        del args, kwargs
        return {"model": {"weight": 1}}


class FakeInnerModel:
    def __init__(self) -> None:
        self.state = None

    def load_state_dict(self, state):
        self.state = state


class FakeRFDETRModel:
    def __init__(self) -> None:
        self.model = FakeInnerModel()

    def predict(self, image, threshold):
        del image, threshold
        return [
            {"bbox": [2, 4, 12, 18], "score": 0.88},
            {"bbox": [10, 10, 28, 26], "score": 0.67},
        ]


def test_ui_detr_adapter_normalizes_predictions(monkeypatch):
    settings = AppSettings(active_model="ui_detr_1", models=default_models())
    detector = UIDetrDetector("ui_detr_1", settings.models["ui_detr_1"], settings)

    monkeypatch.setattr(detector, "resolve_weight_path", lambda: "model.pth")
    monkeypatch.setattr(detector, "_import_components", lambda: (FakeTorch, FakeRFDETRModel))

    detector.load()
    result = detector.predict(image=object())

    assert result.detections[0].label == "interactive_element"
    assert result.detections[0].confidence == 0.88
