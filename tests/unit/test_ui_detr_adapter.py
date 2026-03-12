from __future__ import annotations

from gui_detector_api.detectors.ui_detr import UIDetrDetector
from gui_detector_api.settings import AppSettings, default_models


class FakeDetections:
    xyxy = [[2, 4, 12, 18], [10, 10, 28, 26]]
    confidence = [0.88, 0.67]
    class_id = [1, None]


class FakeRFDETRModel:
    class_names = {1: "button"}

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def predict(self, image, threshold):
        del image, threshold
        return FakeDetections()


def test_ui_detr_adapter_normalizes_predictions(monkeypatch):
    settings = AppSettings(active_model="ui_detr_1", models=default_models())
    settings.models["ui_detr_1"].image_size = 1600
    detector = UIDetrDetector("ui_detr_1", settings.models["ui_detr_1"], settings)

    monkeypatch.setattr(detector, "resolve_weight_path", lambda: "model.pth")
    monkeypatch.setattr(detector, "resolve_device", lambda: "mps")
    monkeypatch.setattr(detector, "_import_model_class", lambda: (lambda **kwargs: FakeRFDETRModel(**kwargs)))

    detector.load()
    result = detector.predict(image=object())

    assert detector._model.kwargs["pretrain_weights"] == "model.pth"
    assert detector._model.kwargs["device"] == "mps"
    assert detector._model.kwargs["resolution"] == 1600
    assert result.detections[0].label == "button"
    assert result.detections[0].confidence == 0.88
    assert result.detections[1].label == "interactive_element"
