from __future__ import annotations

from gui_detector_api.domain.schemas import (
    BoundingBox,
    ClassifiedDetection,
    ClassificationSummary,
    Detection,
    DetectorBackend,
    ImageMetadata,
    ModelMetadata,
    PredictionResponse,
)


def test_prediction_response_serializes_to_expected_shape():
    response = PredictionResponse(
        model=ModelMetadata(
            key="gpa_gui_detector",
            backend=DetectorBackend.ULTRALYTICS,
            hf_repo_id="Salesforce/GPA-GUI-Detector",
        ),
        image=ImageMetadata(filename="sample.png", content_type="image/png", width=10, height=10, size_bytes=20),
        detections=[
            Detection(
                id="det-0001",
                label="button",
                class_id=1,
                confidence=0.99,
                bbox=BoundingBox(x_min=1, y_min=2, x_max=3, y_max=4),
            )
        ],
        classified_detections=[
            ClassifiedDetection(
                id="det-0001",
                label="button",
                class_id=1,
                confidence=0.99,
                bbox=BoundingBox(x_min=1, y_min=2, x_max=3, y_max=4),
                predicted_class="primary_button",
                similarity_score=0.83,
            )
        ],
        classification=ClassificationSummary(applied=True, class_count=2, similarity_threshold=0.3),
    )
    payload = response.model_dump(mode="json")
    assert payload["model"]["backend"] == "ultralytics"
    assert payload["detections"][0]["bbox"]["x_max"] == 3.0
    assert payload["classified_detections"][0]["predicted_class"] == "primary_button"
    assert payload["classification"]["applied"] is True
