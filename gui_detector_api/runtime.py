from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from gui_detector_api.detectors.base import Detector
from gui_detector_api.detectors.model_registry import ModelRegistry
from gui_detector_api.rendering.preview import PreviewRenderer
from gui_detector_api.services.prediction import PredictionService
from gui_detector_api.settings import AppSettings


@dataclass
class RuntimeState:
    settings: AppSettings
    model_registry: ModelRegistry
    detector: Detector | None = None
    prediction_service: PredictionService = field(init=False)
    preview_renderer: PreviewRenderer = field(default_factory=PreviewRenderer)
    ready: bool = False
    load_error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.prediction_service = PredictionService(settings=self.settings, detector=self.detector)

    def mark_ready(self, detector: Detector) -> None:
        self.detector = detector
        self.prediction_service.set_detector(detector)
        self.ready = True
        self.load_error = None

    def mark_failed(self, detail: str) -> None:
        self.detector = None
        self.prediction_service.set_detector(None)
        self.ready = False
        self.load_error = detail
