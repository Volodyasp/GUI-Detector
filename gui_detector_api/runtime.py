from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from gui_detector_api.detectors.base import Detector
from gui_detector_api.detectors.factory import DetectorFactory
from gui_detector_api.settings import AppSettings


@dataclass
class RuntimeState:
    settings: AppSettings
    detector_factory: DetectorFactory
    detector: Detector | None = None
    ready: bool = False
    load_error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def mark_ready(self, detector: Detector) -> None:
        self.detector = detector
        self.ready = True
        self.load_error = None

    def mark_failed(self, detail: str) -> None:
        self.detector = None
        self.ready = False
        self.load_error = detail
