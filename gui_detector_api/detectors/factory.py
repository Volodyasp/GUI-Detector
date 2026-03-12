from __future__ import annotations

from collections.abc import Callable, Mapping

from gui_detector_api.detectors.base import Detector, ModelLoadError
from gui_detector_api.detectors.gpa import GPAUltralyticsDetector
from gui_detector_api.detectors.ui_detr import UIDetrDetector
from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.settings import AppSettings, ModelSettings

DetectorBuilder = Callable[[str, ModelSettings, AppSettings], Detector]


class DetectorFactory:
    def __init__(self, registry: Mapping[DetectorBackend, DetectorBuilder] | None = None) -> None:
        self._registry: dict[DetectorBackend, DetectorBuilder] = {
            DetectorBackend.ULTRALYTICS: lambda model_key, model_settings, app_settings: GPAUltralyticsDetector(
                model_key, model_settings, app_settings
            ),
            DetectorBackend.RFDETR: lambda model_key, model_settings, app_settings: UIDetrDetector(
                model_key, model_settings, app_settings
            ),
        }
        if registry:
            self._registry.update(registry)

    def create(self, settings: AppSettings) -> Detector:
        model_key = settings.active_model
        model_settings = settings.models.get(model_key)
        if model_settings is None:
            raise ModelLoadError(f"Active model '{model_key}' is not defined in settings.")

        builder = self._registry.get(model_settings.backend)
        if builder is None:
            raise ModelLoadError(f"No detector builder is registered for backend '{model_settings.backend}'.")

        return builder(model_key, model_settings, settings)
