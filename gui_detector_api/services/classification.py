from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from PIL import Image

from gui_detector_api.domain.schemas import ClassifiedDetection, ClassificationSummary, Detection
from gui_detector_api.services.class_registry import ClassExemplar, ClassRegistryService
from gui_detector_api.services.embeddings import CLIPEmbeddingService


@dataclass(slots=True)
class ClassificationDecision:
    class_name: str
    score: float


class DetectionClassificationService:
    def __init__(
        self,
        *,
        settings,
        embedding_service: CLIPEmbeddingService,
        class_registry: ClassRegistryService,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.class_registry = class_registry

    def classify_detections(
        self,
        image: Image.Image,
        detections: list[Detection],
    ) -> tuple[list[ClassifiedDetection], ClassificationSummary]:
        summary = ClassificationSummary(
            applied=self.class_registry.has_classes(),
            class_count=self.class_registry.class_count(),
            knn_k=self.settings.classification_knn_k,
            similarity_threshold=self.settings.classification_similarity_threshold,
        )
        if not summary.applied or not detections:
            return [], summary

        exemplars = self.class_registry.get_exemplar_index()
        if not exemplars:
            return [], summary

        valid_detections: list[Detection] = []
        crops: list[Image.Image] = []
        for detection in detections:
            crop = self._crop_detection(image, detection)
            if crop is None:
                continue
            valid_detections.append(detection)
            crops.append(crop)

        if not crops:
            return [], summary

        crop_embeddings = self.embedding_service.embed_images(crops)
        classified_detections: list[ClassifiedDetection] = []
        for detection, embedding in zip(valid_detections, crop_embeddings, strict=False):
            decision = self._classify_embedding(embedding, exemplars)
            if decision is None:
                continue
            classified_detections.append(
                ClassifiedDetection(
                    **detection.model_dump(),
                    predicted_class=decision.class_name,
                    similarity_score=decision.score,
                )
            )

        return classified_detections, summary

    def _crop_detection(self, image: Image.Image, detection: Detection) -> Image.Image | None:
        x_min = max(0, min(image.width, int(round(detection.bbox.x_min))))
        y_min = max(0, min(image.height, int(round(detection.bbox.y_min))))
        x_max = max(0, min(image.width, int(round(detection.bbox.x_max))))
        y_max = max(0, min(image.height, int(round(detection.bbox.y_max))))

        if x_max <= x_min or y_max <= y_min:
            return None
        return image.crop((x_min, y_min, x_max, y_max))

    def _classify_embedding(
        self,
        embedding: list[float],
        exemplars: list[ClassExemplar],
    ) -> ClassificationDecision | None:
        similarities = [
            (self._dot_product(embedding, exemplar.embedding), exemplar)
            for exemplar in exemplars
        ]
        similarities.sort(key=lambda item: item[0], reverse=True)
        top_k = similarities[: max(1, min(self.settings.classification_knn_k, len(similarities)))]

        scores_by_class: dict[str, list[float]] = defaultdict(list)
        exemplar_name_by_class: dict[str, str] = {}
        for similarity, exemplar in top_k:
            scores_by_class[exemplar.class_id].append(similarity)
            exemplar_name_by_class[exemplar.class_id] = exemplar.class_name

        best_class_id: str | None = None
        best_score = -1.0
        best_peak = -1.0
        for class_id, scores in scores_by_class.items():
            average_score = sum(scores) / len(scores)
            peak_score = max(scores)
            if average_score > best_score or (average_score == best_score and peak_score > best_peak):
                best_class_id = class_id
                best_score = average_score
                best_peak = peak_score

        if best_class_id is None or best_score < self.settings.classification_similarity_threshold:
            return None

        return ClassificationDecision(
            class_name=exemplar_name_by_class[best_class_id],
            score=best_score,
        )

    def _dot_product(self, left: list[float], right: list[float]) -> float:
        return float(sum(left_item * right_item for left_item, right_item in zip(left, right, strict=False)))
