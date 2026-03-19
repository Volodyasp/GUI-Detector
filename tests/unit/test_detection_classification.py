from __future__ import annotations

from PIL import Image

from gui_detector_api.domain.schemas import BoundingBox, Detection
from gui_detector_api.services.class_registry import ImageExemplar
from gui_detector_api.services.classification import DetectionClassificationService
from gui_detector_api.services.ocr import OcrBackend, OcrResult, OcrService
from gui_detector_api.settings import AppSettings, default_models


class FakeClassRegistry:
    def __init__(self, text_exemplars=None, embedding_exemplars=None):
        self._text_exemplars = text_exemplars or []
        self._embedding_exemplars = embedding_exemplars or []

    def has_classes(self):
        return bool(self._text_exemplars or self._embedding_exemplars)

    def class_count(self):
        ids = {e[0] for e in self._text_exemplars}
        ids |= {e.class_id for e in self._embedding_exemplars}
        return len(ids)

    def get_text_exemplars(self):
        return list(self._text_exemplars)

    def get_all_exemplars(self):
        return list(self._embedding_exemplars)


class FakeEmbeddingService:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_images(self, images):
        del images
        return list(self.embeddings)


class FakeOcrService:
    def __init__(self, result: OcrResult | None = None):
        self._result = result or OcrResult(text="", confidence=0.0, raw_texts=[])

    def extract_text(self, image):
        del image
        return self._result


def _make_settings(tmp_path, **overrides):
    settings = AppSettings(
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _make_detection(det_id="det-0001", x1=1, y1=1, x2=50, y2=30, confidence=0.9):
    return Detection(
        id=det_id,
        label="element",
        class_id=None,
        confidence=confidence,
        bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
    )


def test_ocr_match_succeeds(tmp_path):
    settings = _make_settings(tmp_path, text_match_threshold=0.65, ocr_confidence_threshold=0.3)
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([]),
        class_registry=FakeClassRegistry(
            text_exemplars=[("c1", "CartButton", "add to cart")],
        ),
        ocr_service=FakeOcrService(OcrResult(text="add to cart", confidence=0.9, raw_texts=["add to cart"])),
    )

    image = Image.new("RGB", (100, 60), "white")
    updated, classified, summary = classifier.classify_detections(image, [_make_detection()])

    assert summary.applied is True
    assert len(classified) == 1
    assert classified[0].predicted_class == "CartButton"
    assert classified[0].match_method == "ocr"
    assert classified[0].similarity_score >= 0.65
    assert updated[0].ocr_text == "add to cart"


def test_ocr_fails_visual_fallback_succeeds(tmp_path):
    settings = _make_settings(
        tmp_path,
        classification_similarity_threshold=0.45,
        ocr_confidence_threshold=0.3,
    )
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([[1.0, 0.0]]),
        class_registry=FakeClassRegistry(
            embedding_exemplars=[
                ImageExemplar(class_id="c1", class_name="BuyButton", embedding=[1.0, 0.0]),
            ],
        ),
        ocr_service=FakeOcrService(OcrResult(text="", confidence=0.0, raw_texts=[])),
    )

    image = Image.new("RGB", (100, 60), "white")
    _, classified, _ = classifier.classify_detections(image, [_make_detection()])

    assert len(classified) == 1
    assert classified[0].predicted_class == "BuyButton"
    assert classified[0].match_method == "visual"


def test_both_fail_returns_unclassified(tmp_path):
    settings = _make_settings(
        tmp_path,
        classification_similarity_threshold=0.95,
        ocr_confidence_threshold=0.3,
        text_match_threshold=0.9,
    )
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([[0.5, 0.5]]),
        class_registry=FakeClassRegistry(
            text_exemplars=[("c1", "CartButton", "add to cart")],
            embedding_exemplars=[
                ImageExemplar(class_id="c1", class_name="CartButton", embedding=[0.0, 1.0]),
            ],
        ),
        ocr_service=FakeOcrService(OcrResult(text="random text", confidence=0.9, raw_texts=["random text"])),
    )

    image = Image.new("RGB", (100, 60), "white")
    _, classified, _ = classifier.classify_detections(image, [_make_detection()])

    assert len(classified) == 0


def test_ocr_disambiguates_multiple_classes(tmp_path):
    settings = _make_settings(tmp_path, text_match_threshold=0.65, ocr_confidence_threshold=0.3)
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([]),
        class_registry=FakeClassRegistry(
            text_exemplars=[
                ("c1", "CartButton", "add to cart"),
                ("c2", "BuyButton", "buy now"),
            ],
        ),
        ocr_service=FakeOcrService(OcrResult(text="buy now", confidence=0.95, raw_texts=["buy now"])),
    )

    image = Image.new("RGB", (100, 60), "white")
    _, classified, _ = classifier.classify_detections(image, [_make_detection()])

    assert len(classified) == 1
    assert classified[0].predicted_class == "BuyButton"


def test_text_only_class_uses_ocr_only_path(tmp_path):
    settings = _make_settings(tmp_path, text_match_threshold=0.65, ocr_confidence_threshold=0.3)
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([]),
        class_registry=FakeClassRegistry(
            text_exemplars=[("c1", "CartButton", "add to cart")],
        ),
        ocr_service=FakeOcrService(OcrResult(text="add to cart", confidence=0.85, raw_texts=["add to cart"])),
    )

    image = Image.new("RGB", (100, 60), "white")
    _, classified, _ = classifier.classify_detections(image, [_make_detection()])

    assert len(classified) == 1
    assert classified[0].predicted_class == "CartButton"
    assert classified[0].match_method == "ocr"


def test_ocr_confidence_below_threshold_falls_to_visual(tmp_path):
    settings = _make_settings(
        tmp_path,
        ocr_confidence_threshold=0.8,
        classification_similarity_threshold=0.45,
    )
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([[1.0, 0.0]]),
        class_registry=FakeClassRegistry(
            text_exemplars=[("c1", "CartButton", "add to cart")],
            embedding_exemplars=[
                ImageExemplar(class_id="c1", class_name="CartButton", embedding=[1.0, 0.0]),
            ],
        ),
        ocr_service=FakeOcrService(OcrResult(text="add to cart", confidence=0.3, raw_texts=["add to cart"])),
    )

    image = Image.new("RGB", (100, 60), "white")
    _, classified, _ = classifier.classify_detections(image, [_make_detection()])

    assert len(classified) == 1
    assert classified[0].match_method == "visual"


def test_no_classes_returns_empty(tmp_path):
    settings = _make_settings(tmp_path)
    classifier = DetectionClassificationService(
        settings=settings,
        embedding_service=FakeEmbeddingService([]),
        class_registry=FakeClassRegistry(),
        ocr_service=FakeOcrService(),
    )

    image = Image.new("RGB", (100, 60), "white")
    _, classified, summary = classifier.classify_detections(image, [_make_detection()])

    assert summary.applied is False
    assert len(classified) == 0
