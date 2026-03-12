from __future__ import annotations

from collections.abc import Mapping

from gui_detector_api.detectors.base import Detector
from gui_detector_api.detectors.model_registry import DetectorBuilder, ModelRegistry
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.settings import AppSettings


class DetectorFactory:
    def __init__(
        self,
        registry: Mapping[DetectorBackend, DetectorBuilder] | ModelRegistry | None = None,
    ) -> None:
        if isinstance(registry, ModelRegistry):
            self.model_registry = registry
        else:
            self.model_registry = ModelRegistry(builders=registry)

    def create(self, settings: AppSettings) -> Detector:
        return self.model_registry.create_active_detector(settings)
