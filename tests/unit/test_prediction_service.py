from __future__ import annotations

import asyncio
from io import BytesIO

from starlette.datastructures import UploadFile

from gui_detector_api.detectors.base import Detector, PredictionInputs
from gui_detector_api.domain.schemas import BoundingBox, ClassificationSummary, Detection, PredictionResult
from gui_detector_api.services.prediction import PredictionService
from gui_detector_api.settings import AppSettings, default_models


class StubDetector(Detector):
    def load(self) -> None:
        return None

    def predict(self, inputs: PredictionInputs) -> PredictionResult:
        del inputs
        return PredictionResult(
            detections=[
                Detection(
                    id="custom-id-2",
                    label="low-confidence",
                    class_id=None,
                    confidence=0.25,
                    bbox=BoundingBox(x_min=1, y_min=1, x_max=10, y_max=10),
                ),
                Detection(
                    id="custom-id-1",
                    label="high-confidence",
                    class_id=1,
                    confidence=0.95,
                    bbox=BoundingBox(x_min=2, y_min=2, x_max=20, y_max=20),
                ),
            ]
        )


class StubClassificationService:
    def classify_detections(self, image, detections):
        del image
        return [], ClassificationSummary(applied=False, class_count=0, knn_k=3, similarity_threshold=0.3)


def test_prediction_service_preserves_adapter_output_order_and_ids(png_bytes, tmp_path):
    settings = AppSettings(
        active_model="gpa_gui_detector",
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    service = PredictionService(settings=settings, classification_service=StubClassificationService())
    detector = StubDetector("gpa_gui_detector", settings.models["gpa_gui_detector"], settings)
    upload = UploadFile(filename="sample.png", file=BytesIO(png_bytes), headers={"content-type": "image/png"})

    response = asyncio.run(service.predict_upload(detector, upload))

    assert [item["id"] for item in response.model_dump()["detections"]] == ["custom-id-2", "custom-id-1"]
    assert [item["label"] for item in response.model_dump()["detections"]] == ["low-confidence", "high-confidence"]
    assert response.classification.applied is False
    assert response.classified_detections == []
