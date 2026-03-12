from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from gui_detector_api.detectors.base import Detector
from gui_detector_api.services.prediction import PredictionService
from gui_detector_api.settings import AppSettings


@dataclass
class RuntimeState:
    settings: AppSettings
    detector: Detector | None = None
    prediction_service: PredictionService = field(init=False)
    load_error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.prediction_service = PredictionService(settings=self.settings)

    @property
    def ready(self) -> bool:
        return self.detector is not None and self.load_error is None

    def mark_ready(self, detector: Detector) -> None:
        self.detector = detector
        self.load_error = None

    def mark_failed(self, detail: str) -> None:
        self.detector = None
        self.load_error = detail
