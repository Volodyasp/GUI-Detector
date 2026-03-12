from __future__ import annotations

from typing import Any

from PIL import Image

from gui_detector_api.detectors.base import (
    Detector,
    DetectorPredictionError,
    ModelLoadError,
    make_detection,
    normalize_result,
    to_list,
)
from gui_detector_api.domain.schemas import PredictionResult


class UIDetrDetector(Detector):
    def __init__(self, model_key, model_settings, app_settings) -> None:
        super().__init__(model_key, model_settings, app_settings)
        self._model = None

    def _import_components(self):
        try:
            import torch
            from rfdetr.detr import RFDETRMedium
        except ImportError as exc:
            raise ModelLoadError(
                "rfdetr is not installed. Run `poetry install --with models` to enable UI-DETR inference."
            ) from exc
        return torch, RFDETRMedium

    def load(self) -> None:
        weight_path = self.resolve_weight_path()
        torch, model_class = self._import_components()
        try:
            model = model_class()
            state = torch.load(str(weight_path), map_location=self.model_settings.device, weights_only=True)
            weights = state.get("model", state)
            model.model.load_state_dict(weights)
            self._model = model
        except Exception as exc:  # pragma: no cover - exercised via tests through injected fakes
            raise ModelLoadError(f"Failed to initialize UI-DETR detector: {exc}") from exc

    def predict(self, image: Image.Image) -> PredictionResult:
        if self._model is None:
            raise DetectorPredictionError("UI-DETR detector has not been loaded.")

        try:
            results = self._model.predict(image, threshold=self.model_settings.confidence_threshold)
        except Exception as exc:
            raise DetectorPredictionError(f"UI-DETR prediction failed: {exc}") from exc

        return self._normalize(results)

    def _normalize(self, raw_predictions: Any) -> PredictionResult:
        detections = []

        if isinstance(raw_predictions, dict):
            boxes = to_list(raw_predictions.get("boxes"))
            scores = to_list(raw_predictions.get("scores"))
            labels = to_list(raw_predictions.get("labels"))
            for index, bbox in enumerate(boxes, start=1):
                label = labels[index - 1] if index - 1 < len(labels) else "interactive_element"
                confidence = scores[index - 1] if index - 1 < len(scores) else 0.0
                detections.append(
                    make_detection(
                        index,
                        label=str(label) or "interactive_element",
                        confidence=confidence,
                        bbox=tuple(float(value) for value in bbox),
                    )
                )
        else:
            items = raw_predictions if isinstance(raw_predictions, list) else [raw_predictions]
            for index, item in enumerate(items, start=1):
                if isinstance(item, dict):
                    bbox = item.get("bbox") or item.get("box") or item.get("xyxy") or (0, 0, 0, 0)
                    confidence = item.get("score") or item.get("confidence") or 0.0
                    label = item.get("label") or "interactive_element"
                else:
                    bbox = getattr(item, "bbox", None) or getattr(item, "box", None) or getattr(item, "xyxy", None)
                    confidence = getattr(item, "score", None) or getattr(item, "confidence", None) or 0.0
                    label = getattr(item, "label", None) or "interactive_element"
                detections.append(
                    make_detection(
                        index,
                        label=str(label) or "interactive_element",
                        confidence=confidence,
                        bbox=tuple(float(value) for value in bbox),
                    )
                )

        return normalize_result(detections)
