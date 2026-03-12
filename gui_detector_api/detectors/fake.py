from __future__ import annotations

from PIL import Image

from gui_detector_api.detectors.base import Detector, ModelLoadError, normalize_result
from gui_detector_api.domain.schemas import Detection, PredictionResult


class FakeDetector(Detector):
    def __init__(
        self,
        model_key,
        model_settings,
        app_settings,
        *,
        detections: list[Detection] | None = None,
        load_error: str | None = None,
        predict_error: str | None = None,
    ) -> None:
        super().__init__(model_key, model_settings, app_settings)
        self._detections = detections or []
        self._load_error = load_error
        self._predict_error = predict_error
        self.loaded = False

    def load(self) -> None:
        if self._load_error:
            raise ModelLoadError(self._load_error)
        self.loaded = True

    def predict(self, image: Image.Image) -> PredictionResult:
        del image
        if self._predict_error:
            raise RuntimeError(self._predict_error)
        return normalize_result(self._detections)
