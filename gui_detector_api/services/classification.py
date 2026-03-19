from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from PIL import Image

from gui_detector_api.domain.schemas import ClassifiedDetection, ClassificationSummary, Detection
from gui_detector_api.services.class_registry import ClassRegistryService, ImageExemplar
from gui_detector_api.services.embeddings import EmbeddingService
from gui_detector_api.services.ocr import OcrService
from gui_detector_api.services.text_matching import match_text_against_exemplars
from gui_detector_api.settings import AppSettings


@dataclass(slots=True)
class ClassificationDecision:
    class_name: str
    score: float
    match_method: str


class DetectionClassificationService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        embedding_service: EmbeddingService,
        class_registry: ClassRegistryService,
        ocr_service: OcrService,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.class_registry = class_registry
        self.ocr_service = ocr_service

    def classify_detections(
        self,
        image: Image.Image,
        detections: list[Detection],
    ) -> tuple[list[Detection], list[ClassifiedDetection], ClassificationSummary]:
        summary = ClassificationSummary(
            applied=self.class_registry.has_classes(),
            class_count=self.class_registry.class_count(),
            similarity_threshold=self.settings.classification_similarity_threshold,
        )
        if not detections:
            return detections, [], summary

        # Run OCR on all crops and annotate detections with ocr_text
        updated_detections: list[Detection] = []
        valid: list[tuple[Detection, Image.Image]] = []
        for detection in detections:
            crop = self._crop_detection(image, detection)
            if crop is None:
                updated_detections.append(detection)
                continue
            ocr_result = self.ocr_service.extract_text(crop)
            ocr_text = ocr_result.text if ocr_result.text and ocr_result.confidence >= self.settings.ocr_confidence_threshold else None
            annotated = detection.model_copy(update={"ocr_text": ocr_text})
            updated_detections.append(annotated)
            valid.append((annotated, crop))

        if not summary.applied:
            return updated_detections, [], summary

        text_exemplars = self.class_registry.get_text_exemplars()
        embedding_exemplars = self.class_registry.get_all_exemplars()

        if not text_exemplars and not embedding_exemplars:
            return updated_detections, [], summary

        if not valid:
            return updated_detections, [], summary

        # Phase 1: Try OCR-based classification
        classified: list[ClassifiedDetection] = []
        needs_visual: list[tuple[Detection, Image.Image]] = []

        for detection, crop in valid:
            if detection.ocr_text and text_exemplars:
                match = match_text_against_exemplars(
                    detection.ocr_text,
                    text_exemplars,
                    threshold=self.settings.text_match_threshold,
                )
                if match is not None:
                    classified.append(
                        ClassifiedDetection(
                            **detection.model_dump(),
                            predicted_class=match.class_name,
                            similarity_score=match.score,
                            match_method="ocr",
                        )
                    )
                    continue
            needs_visual.append((detection, crop))

        # Phase 2: Embedding fallback (matches crop against text + image exemplar embeddings)
        if needs_visual and embedding_exemplars:
            crops_for_embedding = [crop for _, crop in needs_visual]
            crop_embeddings = self.embedding_service.embed_images(crops_for_embedding)

            for (detection, _), embedding in zip(needs_visual, crop_embeddings, strict=False):
                decision = self._classify_by_visual_similarity(embedding, embedding_exemplars)
                if decision is not None:
                    classified.append(
                        ClassifiedDetection(
                            **detection.model_dump(),
                            predicted_class=decision.class_name,
                            similarity_score=decision.score,
                            match_method="visual",
                        )
                    )

        return updated_detections, classified, summary

    def _crop_detection(self, image: Image.Image, detection: Detection) -> Image.Image | None:
        x_min = max(0, min(image.width, int(round(detection.bbox.x_min))))
        y_min = max(0, min(image.height, int(round(detection.bbox.y_min))))
        x_max = max(0, min(image.width, int(round(detection.bbox.x_max))))
        y_max = max(0, min(image.height, int(round(detection.bbox.y_max))))

        if x_max <= x_min or y_max <= y_min:
            return None
        return image.crop((x_min, y_min, x_max, y_max))

    def _classify_by_visual_similarity(
        self,
        embedding: list[float],
        image_exemplars: list[ImageExemplar],
    ) -> ClassificationDecision | None:
        scores_by_class: dict[str, list[float]] = defaultdict(list)
        name_by_class: dict[str, str] = {}

        for exemplar in image_exemplars:
            similarity = self._dot_product(embedding, exemplar.embedding)
            scores_by_class[exemplar.class_id].append(similarity)
            name_by_class[exemplar.class_id] = exemplar.class_name

        ranked: list[tuple[str, float]] = []
        for class_id, scores in scores_by_class.items():
            scores.sort(reverse=True)
            top_k = scores[: max(1, min(self.settings.classification_knn_k, len(scores)))]
            avg = sum(top_k) / len(top_k)
            ranked.append((class_id, avg))

        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return None

        best_id, best_score = ranked[0]
        if best_score < self.settings.classification_similarity_threshold:
            return None

        if len(ranked) > 1:
            _, second_score = ranked[1]
            if best_score - second_score < self.settings.classification_similarity_margin:
                return None

        return ClassificationDecision(
            class_name=name_by_class[best_id],
            score=best_score,
            match_method="visual",
        )

    def _dot_product(self, left: list[float], right: list[float]) -> float:
        return float(sum(a * b for a, b in zip(left, right, strict=False)))
