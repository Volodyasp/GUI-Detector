from __future__ import annotations

from typing import Any

from gui_detector_api.detectors.base import (
    Detector,
    DetectorPredictionError,
    ModelLoadError,
    PredictionInputs,
    make_detection,
    normalize_result,
    to_list,
)
from gui_detector_api.domain.schemas import PredictionResult
from gui_detector_api.settings import AppSettings, ModelSettings


class UIDetrDetector(Detector):
    def __init__(self, model_key: str, model_settings: ModelSettings, app_settings: AppSettings) -> None:
        super().__init__(model_key, model_settings, app_settings)
        self._model = None

    def _import_model_class(self):
        try:
            from rfdetr.detr import RFDETRMedium
        except ImportError as exc:
            raise ModelLoadError(
                "rfdetr is not installed. Run `poetry install --with models` to enable UI-DETR inference."
            ) from exc
        return RFDETRMedium

    def load(self) -> None:
        weight_path = self.resolve_weight_path()
        model_class = self._import_model_class()
        resolved_device = self.resolve_device()
        kwargs: dict[str, Any] = {
            "pretrain_weights": str(weight_path),
            "device": resolved_device,
        }
        if self.model_settings.image_size is not None:
            kwargs["resolution"] = self.model_settings.image_size
        try:
            self._model = model_class(**kwargs)
            self.resolved_device = resolved_device
        except Exception as exc:  # pragma: no cover - exercised via tests through injected fakes
            raise ModelLoadError(f"Failed to initialize UI-DETR detector: {exc}") from exc

    def predict(self, inputs: PredictionInputs) -> PredictionResult:
        if self._model is None:
            raise DetectorPredictionError("UI-DETR detector has not been loaded.")

        try:
            results = self._model.predict(inputs.image, threshold=self.model_settings.confidence_threshold)
        except Exception as exc:
            raise DetectorPredictionError(f"UI-DETR prediction failed: {exc}") from exc

        return self._normalize(results)

    def _normalize(self, raw_predictions: Any) -> PredictionResult:
        detections = []
        items = raw_predictions if isinstance(raw_predictions, list) else [raw_predictions]
        for item in items:
            detections.extend(self._normalize_item(item, start_index=len(detections) + 1))
        return normalize_result(detections)

    def _normalize_item(self, item: Any, *, start_index: int) -> list:
        if item is None:
            return []
        if isinstance(item, dict):
            return self._normalize_dict_item(item, start_index=start_index)
        return self._normalize_detections_like(item, start_index=start_index)

    def _normalize_dict_item(self, item: dict[str, Any], *, start_index: int) -> list:
        boxes = to_list(item.get("boxes"))
        scores = to_list(item.get("scores"))
        labels = to_list(item.get("labels"))
        if boxes:
            detections = []
            for index, bbox in enumerate(boxes, start=start_index):
                offset = index - start_index
                label = labels[offset] if offset < len(labels) else "interactive_element"
                confidence = scores[offset] if offset < len(scores) else 0.0
                detections.append(
                    make_detection(
                        index,
                        label=str(label) or "interactive_element",
                        confidence=confidence,
                        bbox=tuple(float(value) for value in bbox),
                    )
                )
            return detections

        bbox = item.get("bbox") or item.get("box") or item.get("xyxy") or (0, 0, 0, 0)
        confidence = item.get("score") or item.get("confidence") or 0.0
        label = item.get("label") or "interactive_element"
        return [
            make_detection(
                start_index,
                label=str(label) or "interactive_element",
                confidence=confidence,
                bbox=tuple(float(value) for value in bbox),
            )
        ]

    def _normalize_detections_like(self, item: Any, *, start_index: int) -> list:
        coordinates = self._coerce_boxes(to_list(getattr(item, "xyxy", [])))
        if not coordinates:
            bbox = getattr(item, "bbox", None) or getattr(item, "box", None)
            if bbox is not None:
                coordinates = [tuple(float(value) for value in to_list(bbox))]
        confidences = to_list(getattr(item, "confidence", [])) or to_list(getattr(item, "scores", []))
        class_ids = to_list(getattr(item, "class_id", [])) or to_list(getattr(item, "class_ids", []))
        label_map = self._get_label_map(item)

        detections = []
        for index, bbox in enumerate(coordinates, start=start_index):
            offset = index - start_index
            class_id = self._coerce_class_id(class_ids[offset] if offset < len(class_ids) else None)
            confidence = confidences[offset] if offset < len(confidences) else 0.0
            detections.append(
                make_detection(
                    index,
                    label=self._resolve_label(class_id, label_map),
                    confidence=confidence,
                    bbox=tuple(float(value) for value in bbox),
                    class_id=class_id,
                )
            )
        return detections

    def _get_label_map(self, item: Any) -> dict[int, str]:
        for candidate in (
            getattr(item, "class_names", None),
            getattr(self._model, "class_names", None),
            getattr(self._model, "classes", None),
        ):
            label_map = self._coerce_label_map(candidate)
            if label_map:
                return label_map
        return {}

    def _coerce_label_map(self, raw_label_map: Any) -> dict[int, str]:
        if isinstance(raw_label_map, dict):
            coerced: dict[int, str] = {}
            for key, value in raw_label_map.items():
                try:
                    coerced[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            return coerced
        if isinstance(raw_label_map, (list, tuple)):
            return {index: str(value) for index, value in enumerate(raw_label_map)}
        return {}

    def _resolve_label(self, class_id: int | None, label_map: dict[int, str]) -> str:
        if class_id is None:
            return "interactive_element"
        return str(label_map.get(class_id) or label_map.get(class_id + 1) or "interactive_element")

    def _coerce_class_id(self, class_id: Any) -> int | None:
        if class_id is None:
            return None
        try:
            return int(class_id)
        except (TypeError, ValueError):
            return None

    def _coerce_boxes(self, raw_boxes: list[Any]) -> list[tuple[float, float, float, float]]:
        if not raw_boxes:
            return []

        first_item = raw_boxes[0]
        if isinstance(first_item, (int, float)) and len(raw_boxes) == 4:
            return [tuple(float(value) for value in raw_boxes)]

        boxes: list[tuple[float, float, float, float]] = []
        for raw_box in raw_boxes:
            values = to_list(raw_box)
            if len(values) != 4:
                continue
            boxes.append(tuple(float(value) for value in values))
        return boxes
