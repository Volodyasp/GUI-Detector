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


class GPAUltralyticsDetector(Detector):
    def __init__(self, model_key, model_settings, app_settings) -> None:
        super().__init__(model_key, model_settings, app_settings)
        self._model = None

    def _import_yolo_class(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "ultralytics is not installed. Run `poetry install --with models` to enable GPA inference."
            ) from exc
        return YOLO

    def load(self) -> None:
        weight_path = self.resolve_weight_path()
        yolo_class = self._import_yolo_class()
        try:
            resolved_device = self.resolve_device()
            self._model = yolo_class(str(weight_path))
            self.resolved_device = resolved_device
        except Exception as exc:  # pragma: no cover - exercised via tests through injected fakes
            raise ModelLoadError(f"Failed to initialize GPA detector: {exc}") from exc

    def predict(self, image: Image.Image) -> PredictionResult:
        if self._model is None:
            raise DetectorPredictionError("GPA detector has not been loaded.")

        kwargs: dict[str, Any] = {
            "source": image,
            "conf": self.model_settings.confidence_threshold,
            "verbose": False,
        }
        if self.resolved_device:
            kwargs["device"] = self.resolved_device
        if self.model_settings.iou_threshold is not None:
            kwargs["iou"] = self.model_settings.iou_threshold
        if self.model_settings.image_size is not None:
            kwargs["imgsz"] = self.model_settings.image_size

        try:
            results = self._model.predict(**kwargs)
        except Exception as exc:
            raise DetectorPredictionError(f"GPA prediction failed: {exc}") from exc

        return self._normalize(results)

    def _normalize(self, results: Any) -> PredictionResult:
        first_result = results[0] if results else None
        if first_result is None or getattr(first_result, "boxes", None) is None:
            return PredictionResult()

        boxes = first_result.boxes
        coordinates = to_list(getattr(boxes, "xyxy", []))
        confidences = to_list(getattr(boxes, "conf", []))
        class_ids = to_list(getattr(boxes, "cls", []))
        names = getattr(first_result, "names", None) or getattr(self._model, "names", {}) or {}

        detections = []
        for index, bbox in enumerate(coordinates, start=1):
            class_id = int(class_ids[index - 1]) if index - 1 < len(class_ids) else None
            label = names.get(class_id, f"class_{class_id}") if class_id is not None else "interactive_element"
            confidence = confidences[index - 1] if index - 1 < len(confidences) else 0.0
            detections.append(
                make_detection(
                    index,
                    label=label,
                    confidence=confidence,
                    bbox=tuple(float(value) for value in bbox),
                    class_id=class_id,
                )
            )

        return normalize_result(detections)
