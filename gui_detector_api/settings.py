from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gui_detector_api.domain.schemas import DetectorBackend
from gui_detector_api.utils.device import DeviceName


class ModelSettings(BaseModel):
    backend: DetectorBackend
    hf_repo_id: str
    weight_filename: str
    device: DeviceName = "auto"
    confidence_threshold: float = 0.3
    iou_threshold: float | None = None
    image_size: int | None = None
    local_files_only: bool = False


def default_models() -> dict[str, ModelSettings]:
    return {
        "gpa_gui_detector": ModelSettings(
            backend=DetectorBackend.ULTRALYTICS,
            hf_repo_id="Salesforce/GPA-GUI-Detector",
            weight_filename="model.pt",
            confidence_threshold=0.35,
            iou_threshold=0.7,
            image_size=1280,
        ),
        "ui_detr_1": ModelSettings(
            backend=DetectorBackend.RFDETR,
            hf_repo_id="racineai/UI-DETR-1",
            weight_filename="model.pth",
            confidence_threshold=0.35,
        ),
    }


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GUI_DETECTOR_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    service_name: str = "gui-detector-api"
    app_version: str = "0.1.0"
    active_model: str = "ui_detr_1"
    models: dict[str, ModelSettings] = Field(default_factory=default_models)
    model_cache_dir: Path = Path("model-cache")
    max_upload_size_bytes: int = 10 * 1024 * 1024
    prediction_timeout_seconds: float = 60.0
    log_level: str = "INFO"


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
