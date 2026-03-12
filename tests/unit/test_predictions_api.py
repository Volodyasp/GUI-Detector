from __future__ import annotations

from fastapi.testclient import TestClient


def test_prediction_endpoint_returns_normalized_json(ready_app, png_bytes):
    with TestClient(ready_app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", png_bytes, "image/png")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]["key"] == "gpa_gui_detector"
    assert payload["detections"][0]["confidence"] >= payload["detections"][1]["confidence"]
    assert "annotated_image" not in payload


def test_prediction_endpoint_rejects_legacy_image_format_parameter(ready_app, png_bytes):
    with TestClient(ready_app) as client:
        response = client.post(
            "/v1/predictions?image_format=base64",
            files={"image": ("sample.png", png_bytes, "image/png")},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "unsupported_parameter"
    assert "image_format" in payload["detail"]


def test_prediction_endpoint_rejects_invalid_content_type(ready_app):
    with TestClient(ready_app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 415
    assert response.json()["error"] == "invalid_upload"


def test_prediction_endpoint_rejects_corrupt_bytes(ready_app):
    with TestClient(ready_app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", b"not-a-real-image", "image/png")},
        )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_upload"


def test_prediction_endpoint_rejects_empty_upload(ready_app):
    with TestClient(ready_app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", b"", "image/png")},
        )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_upload"


def test_prediction_endpoint_returns_503_when_model_is_unavailable(not_ready_app, png_bytes):
    with TestClient(not_ready_app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", png_bytes, "image/png")},
        )
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_prediction_service_is_a_lifespan_singleton(ready_app):
    with TestClient(ready_app) as client:
        runtime = client.app.state.runtime
        first_service = runtime.prediction_service
        client.get("/v1/healthcheck")
        assert client.app.state.runtime.prediction_service is first_service
