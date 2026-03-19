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
    ocr_text: str | None = None


class ClassifiedDetection(Detection):
    predicted_class: str
    similarity_score: float = Field(ge=-1.0, le=1.0)
    match_method: str | None = None


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


class ClassificationSummary(BaseModel):
    applied: bool = False
    class_count: int = 0
    similarity_threshold: float | None = None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    model: ModelMetadata
    image: ImageMetadata
    detections: list[Detection]
    classified_detections: list[ClassifiedDetection] = Field(default_factory=list)
    classification: ClassificationSummary = Field(default_factory=ClassificationSummary)


class UserClassResponse(BaseModel):
    class_id: str
    name: str
    texts: list[str] = Field(default_factory=list)
    image_filenames: list[str] = Field(default_factory=list)
    text_count: int = 0
    image_count: int = 0
    exemplar_count: int = 0


class UserClassesResponse(BaseModel):
    classes: list[UserClassResponse] = Field(default_factory=list)


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
