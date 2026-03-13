from __future__ import annotations

from PIL import Image

from gui_detector_api.domain.schemas import BoundingBox, Detection
from gui_detector_api.services.class_registry import ClassExemplar
from gui_detector_api.services.classification import DetectionClassificationService
from gui_detector_api.settings import AppSettings, default_models


class FakeClassRegistry:
    def __init__(self, exemplars):
        self._exemplars = exemplars

    def has_classes(self):
        return bool(self._exemplars)

    def class_count(self):
        return len({exemplar.class_id for exemplar in self._exemplars})

    def get_exemplar_index(self):
        return list(self._exemplars)


class FakeEmbeddingService:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_images(self, images):
        del images
        return list(self.embeddings)


def test_detection_classifier_keeps_only_thresholded_matches(tmp_path):
    settings = AppSettings(
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    settings.classification_knn_k = 3
    settings.classification_similarity_threshold = 0.95

    exemplars = [
        ClassExemplar(class_id="class-a", class_name="primary_button", embedding=[1.0, 0.0]),
        ClassExemplar(class_id="class-a", class_name="primary_button", embedding=[0.98, 0.02]),
        ClassExemplar(class_id="class-b", class_name="input_field", embedding=[0.0, 1.0]),
    ]
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([[1.0, 0.0], [0.4, 0.6]]),
        class_registry=FakeClassRegistry(exemplars),
    )

    image = Image.new("RGB", (100, 100), "white")
    detections = [
        Detection(
            id="det-0001",
            label="icon",
            class_id=None,
            confidence=0.91,
            bbox=BoundingBox(x_min=1, y_min=1, x_max=20, y_max=20),
        ),
        Detection(
            id="det-0002",
            label="icon",
            class_id=None,
            confidence=0.84,
            bbox=BoundingBox(x_min=25, y_min=25, x_max=50, y_max=50),
        ),
    ]

    classified, summary = classifier.classify_detections(image, detections)

    assert summary.applied is True
    assert summary.class_count == 2
    assert len(classified) == 1
    assert classified[0].id == "det-0001"
    assert classified[0].predicted_class == "primary_button"
    assert classified[0].similarity_score >= 0.95
