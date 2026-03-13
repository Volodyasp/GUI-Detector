from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from gui_detector_api.detectors.base import Detector
from gui_detector_api.services.class_registry import ClassRegistryService
from gui_detector_api.services.classification import DetectionClassificationService
from gui_detector_api.services.embeddings import CLIPEmbeddingService
from gui_detector_api.services.prediction import PredictionService
from gui_detector_api.settings import AppSettings


@dataclass
class RuntimeState:
    settings: AppSettings
    detector: Detector | None = None
    embedding_service: CLIPEmbeddingService = field(init=False)
    class_registry: ClassRegistryService = field(init=False)
    classification_service: DetectionClassificationService = field(init=False)
    prediction_service: PredictionService = field(init=False)
    load_error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.embedding_service = CLIPEmbeddingService(settings=self.settings)
        self.class_registry = ClassRegistryService(settings=self.settings, embedding_service=self.embedding_service)
        self.classification_service = DetectionClassificationService(
            settings=self.settings,
            embedding_service=self.embedding_service,
            class_registry=self.class_registry,
        )
        self.prediction_service = PredictionService(
            settings=self.settings,
            classification_service=self.classification_service,
        )

    @property
    def ready(self) -> bool:
        return self.detector is not None and self.load_error is None

    def mark_ready(self, detector: Detector) -> None:
        self.detector = detector
        self.load_error = None

    def mark_failed(self, detail: str) -> None:
        self.detector = None
        self.load_error = detail
