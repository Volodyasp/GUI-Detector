from __future__ import annotations

from fastapi.testclient import TestClient

from gui_detector_api.domain.schemas import BoundingBox, ClassifiedDetection, ClassificationSummary, Detection


class StubClassificationService:
    def classify_detections(self, image, detections):
        del image
        if not detections:
            return detections, [], ClassificationSummary(applied=True, class_count=1, similarity_threshold=0.3)
        selected = detections[0]
        return detections, [
            ClassifiedDetection(
                **selected.model_dump(),
                predicted_class="primary_button",
                similarity_score=0.88,
            )
        ], ClassificationSummary(applied=True, class_count=2, similarity_threshold=0.3)


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_images(self, images):
        return [[0.0, 1.0] for _ in images]


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
    assert payload["classified_detections"] == []
    assert payload["classification"]["applied"] is False


def test_prediction_endpoint_returns_classified_subset_when_classification_applies(ready_app, png_bytes):
    with TestClient(ready_app) as client:
        client.app.state.runtime.prediction_service.classification_service = StubClassificationService()
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", png_bytes, "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["detections"]) == 2
    assert len(payload["classified_detections"]) == 1
    assert payload["classified_detections"][0]["predicted_class"] == "primary_button"
    assert payload["classification"]["applied"] is True


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


def test_prediction_endpoint_rejects_legacy_reference_inputs(ready_app, png_bytes):
    with TestClient(ready_app) as client:
        response = client.post(
            "/v1/predictions",
            files=[
                ("image", ("sample.png", png_bytes, "image/png")),
                ("query_texts", (None, "button")),
            ],
        )

    assert response.status_code == 400
    assert response.json()["error"] == "unsupported_parameter"


def test_classes_crud_endpoints_accept_texts_and_images(ready_app, png_bytes):
    fake_embedding_service = FakeEmbeddingService()

    with TestClient(ready_app) as client:
        runtime = client.app.state.runtime
        runtime.class_registry.embedding_service = fake_embedding_service

        created = client.post(
            "/v1/classes",
            files=[
                ("name", (None, "primary_button")),
                ("texts", (None, "button")),
                ("texts", (None, "cta")),
                ("images", ("reference.png", png_bytes, "image/png")),
            ],
        )
        assert created.status_code == 200
        payload = created.json()
        class_id = payload["class_id"]
        assert payload["name"] == "primary_button"
        assert payload["text_count"] == 2
        assert payload["image_count"] == 1

        listed = client.get("/v1/classes")
        assert listed.status_code == 200
        assert listed.json()["classes"][0]["class_id"] == class_id

        replaced = client.put(
            f"/v1/classes/{class_id}",
            files=[
                ("name", (None, "secondary_button")),
                ("texts", (None, "secondary action")),
                ("images", ("replacement.png", png_bytes, "image/png")),
            ],
        )
        assert replaced.status_code == 200
        assert replaced.json()["name"] == "secondary_button"
        assert replaced.json()["text_count"] == 1

        deleted = client.delete(f"/v1/classes/{class_id}")
        assert deleted.status_code == 204

        listed_again = client.get("/v1/classes")
        assert listed_again.status_code == 200
        assert listed_again.json()["classes"] == []


def test_classes_endpoint_accepts_texts_only(ready_app):
    with TestClient(ready_app) as client:
        runtime = client.app.state.runtime
        runtime.class_registry.embedding_service = FakeEmbeddingService()
        response = client.post(
            "/v1/classes",
            files=[
                ("name", (None, "text_only")),
                ("texts", (None, "primary button")),
            ],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text_count"] == 1
    assert payload["image_count"] == 0


def test_classes_endpoint_accepts_images_only(ready_app, png_bytes):
    with TestClient(ready_app) as client:
        runtime = client.app.state.runtime
        runtime.class_registry.embedding_service = FakeEmbeddingService()
        response = client.post(
            "/v1/classes",
            files=[
                ("name", (None, "image_only")),
                ("images", ("reference.png", png_bytes, "image/png")),
            ],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text_count"] == 0
    assert payload["image_count"] == 1


def test_classes_endpoint_rejects_empty_definition(ready_app):
    with TestClient(ready_app) as client:
        response = client.post("/v1/classes", data={"name": "empty"})

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_class_definition"


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
        second_service = runtime.prediction_service
    assert first_service is second_service
