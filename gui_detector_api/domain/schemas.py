from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectorBackend(StrEnum):
    RFDETR = "rfdetr"
    ULTRALYTICS = "ultralytics"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    id: str
    label: str
    class_id: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class PredictionResult(BaseModel):
    detections: list[Detection] = Field(default_factory=list)


class ModelMetadata(BaseModel):
    key: str
    backend: DetectorBackend
    hf_repo_id: str


class ImageMetadata(BaseModel):
    filename: str | None = None
    content_type: str
    width: int
    height: int
    size_bytes: int


class PredictionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    model: ModelMetadata
    image: ImageMetadata
    detections: list[Detection]


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    active_model: str
    backend: DetectorBackend | None = None
    detail: str | None = None
