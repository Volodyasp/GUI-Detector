from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import hf_hub_download
from PIL import Image

from gui_detector_api.domain.schemas import BoundingBox, Detection, ModelMetadata, PredictionResult
from gui_detector_api.utils.device import DeviceResolutionError, resolve_device

if TYPE_CHECKING:
    from gui_detector_api.settings import AppSettings, ModelSettings


class DetectorError(Exception):
    """Base detector exception."""


class ModelLoadError(DetectorError):
    """Raised when the detector runtime cannot be initialized."""


class DetectorPredictionError(DetectorError):
    """Raised when prediction execution fails."""


class Detector(ABC):
    def __init__(self, model_key: str, model_settings: ModelSettings, app_settings: AppSettings) -> None:
        self.model_key = model_key
        self.model_settings = model_settings
        self.app_settings = app_settings
        self.resolved_device: str | None = None

    @property
    def info(self) -> ModelMetadata:
        return ModelMetadata(
            key=self.model_key,
            backend=self.model_settings.backend,
            hf_repo_id=self.model_settings.hf_repo_id,
        )

    @abstractmethod
    def load(self) -> None:
        """Load the model runtime into memory."""

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictionResult:
        """Run synchronous prediction against an in-memory image."""

    def resolve_weight_path(self) -> Path:
        configured_path = Path(self.model_settings.weight_filename).expanduser()
        if configured_path.exists():
            return configured_path

        target_dir = self.app_settings.model_cache_dir / self.model_key
        target_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            hf_hub_download(
                repo_id=self.model_settings.hf_repo_id,
                filename=self.model_settings.weight_filename,
                local_dir=target_dir,
                local_files_only=self.model_settings.local_files_only,
            )
        )

    def resolve_device(self) -> str:
        try:
            self.resolved_device = resolve_device(self.model_settings.device)
        except DeviceResolutionError as exc:
            raise ModelLoadError(str(exc)) from exc
        return self.resolved_device


def make_detection(
    index: int,
    *,
    label: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
    class_id: int | None = None,
) -> Detection:
    return Detection(
        id=f"det-{index:04d}",
        label=label,
        class_id=class_id,
        confidence=max(0.0, min(1.0, float(confidence))),
        bbox=BoundingBox(
            x_min=float(bbox[0]),
            y_min=float(bbox[1]),
            x_max=float(bbox[2]),
            y_max=float(bbox[3]),
        ),
    )


def normalize_result(detections: list[Detection]) -> PredictionResult:
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    renumbered = [
        item.model_copy(update={"id": f"det-{index:04d}"})
        for index, item in enumerate(ordered, start=1)
    ]
    return PredictionResult(detections=renumbered)


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)
