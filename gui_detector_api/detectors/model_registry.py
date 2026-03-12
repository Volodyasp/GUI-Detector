from __future__ import annotations

from collections.abc import Callable, Mapping

from gui_detector_api.detectors.base import Detector, ModelLoadError
from gui_detector_api.detectors.gpa import GPAUltralyticsDetector
from gui_detector_api.detectors.ui_detr import UIDetrDetector
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.settings import AppSettings, ModelSettings

DetectorBuilder = Callable[[str, ModelSettings, AppSettings], Detector]


class ModelRegistry:
    def __init__(self, builders: Mapping[DetectorBackend, DetectorBuilder] | None = None) -> None:
        self._builders: dict[DetectorBackend, DetectorBuilder] = {
            DetectorBackend.ULTRALYTICS: lambda model_key, model_settings, app_settings: GPAUltralyticsDetector(
                model_key, model_settings, app_settings
            ),
            DetectorBackend.RFDETR: lambda model_key, model_settings, app_settings: UIDetrDetector(
                model_key, model_settings, app_settings
            ),
        }
        if builders:
            self._builders.update(builders)

    def register(self, backend: DetectorBackend, builder: DetectorBuilder) -> None:
        self._builders[backend] = builder

    def get_builder(self, backend: DetectorBackend) -> DetectorBuilder:
        builder = self._builders.get(backend)
        if builder is None:
            raise ModelLoadError(f"No detector builder is registered for backend '{backend}'.")
        return builder

    def create_active_detector(self, settings: AppSettings) -> Detector:
        model_key = settings.active_model
        model_settings = settings.models.get(model_key)
        if model_settings is None:
            raise ModelLoadError(f"Active model '{model_key}' is not defined in settings.")

        builder = self.get_builder(model_settings.backend)
        return builder(model_key, model_settings, settings)
