from __future__ import annotations

from fastapi.testclient import TestClient
from PIL import Image

from gui_detector_api.domain.schemas import BoundingBox, Detection, DetectorBackend, ImageMetadata, ModelMetadata, PredictionResponse
from gui_detector_api.rendering.preview import PreviewRenderer


def test_preview_renderer_draws_html_output():
    renderer = PreviewRenderer()
    image = Image.new("RGB", (80, 60), "white")
    prediction = PredictionResponse(
        model=ModelMetadata(
            key="ui_detr_1",
            backend=DetectorBackend.RFDETR,
            hf_repo_id="racineai/UI-DETR-1",
        ),
        image=ImageMetadata(filename="sample.png", content_type="image/png", width=80, height=60, size_bytes=20),
        detections=[
            Detection(
                id="det-0001",
                label="interactive_element",
                class_id=None,
                confidence=0.88,
                bbox=BoundingBox(x_min=4, y_min=6, x_max=40, y_max=24),
            )
        ],
    )
    html = renderer.render(prediction, image)
    assert "Prediction Preview" in html
    assert "interactive_element" in html
    assert "data:image/png;base64," in html


def test_preview_endpoint_returns_html_and_matches_json_flow(ready_app, png_bytes):
    with TestClient(ready_app) as client:
        json_response = client.post("/v1/predictions", files={"image": ("sample.png", png_bytes, "image/png")})
        html_response = client.post(
            "/v1/predictions/preview",
            files={"image": ("sample.png", png_bytes, "image/png")},
        )

    assert json_response.status_code == 200
    assert html_response.status_code == 200
    assert html_response.headers["content-type"].startswith("text/html")
    assert json_response.json()["detections"][0]["label"] in html_response.text
    assert json_response.json()["detections"][0]["id"] in html_response.text
