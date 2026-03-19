from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Any

from gui_detector_api.errors import EmbeddingUnavailableError
from gui_detector_api.utils.device import DeviceResolutionError, resolve_device

if TYPE_CHECKING:
    from PIL import Image

    from gui_detector_api.settings import AppSettings


def normalize_vector(values: list[float]) -> list[float]:
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        return values
    return [value / norm for value in values]


class EmbeddingService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._model = None
        self._processor = None
        self._torch = None
        self.resolved_device: str | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        encoded = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.resolved_device)
        with self._lock, self._torch.no_grad():
            features = self._model.get_text_features(**encoded)
        return self._to_normalized_rows(features)

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        self._ensure_loaded()
        encoded = self._processor(images=images, return_tensors="pt").to(self.resolved_device)
        with self._lock, self._torch.no_grad():
            features = self._model.get_image_features(**encoded)
        return self._to_normalized_rows(features)

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None and self._processor is not None and self._torch is not None:
                return

            try:
                ModelClass, ProcessorClass, torch = self._import_runtime()
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    "Embedding model dependencies are not installed. Run `poetry install --with models` to enable them."
                ) from exc

            try:
                self.resolved_device = resolve_device(
                    self.settings.embedding_model.device,
                    torch_module=torch,
                )
            except DeviceResolutionError as exc:
                raise EmbeddingUnavailableError(str(exc)) from exc

            cache_dir = self.settings.model_cache_dir / "embedding-model"
            cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                self._processor = ProcessorClass.from_pretrained(
                    self.settings.embedding_model.hf_repo_id,
                    cache_dir=str(cache_dir),
                    local_files_only=self.settings.embedding_model.local_files_only,
                )
                self._model = ModelClass.from_pretrained(
                    self.settings.embedding_model.hf_repo_id,
                    cache_dir=str(cache_dir),
                    local_files_only=self.settings.embedding_model.local_files_only,
                )
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"Failed to load embedding model '{self.settings.embedding_model.hf_repo_id}': {exc}"
                ) from exc

            self._torch = torch
            self._model.to(self.resolved_device)
            self._model.eval()

    def _to_normalized_rows(self, values: Any) -> list[list[float]]:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            rows = values.tolist()
        else:
            rows = list(values)
        if rows and isinstance(rows[0], (int, float)):
            rows = [rows]
        return [normalize_vector([float(item) for item in row]) for row in rows]

    def _import_runtime(self):
        from transformers import AutoModel, AutoProcessor
        import torch

        return AutoModel, AutoProcessor, torch
