from __future__ import annotations

from gui_detector_api.domain.schemas import BoundingBox, Detection, DetectorBackend, ImageMetadata, ModelMetadata, PredictionResponse


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
    )
    payload = response.model_dump(mode="json")
    assert payload["model"]["backend"] == "ultralytics"
    assert payload["detections"][0]["bbox"]["x_max"] == 3.0
