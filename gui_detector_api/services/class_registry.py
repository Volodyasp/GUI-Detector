from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import UploadFile
from pydantic import BaseModel, Field

from gui_detector_api.domain.schemas import UserClassResponse, UserClassesResponse
from gui_detector_api.errors import ClassNotFoundError, InvalidClassDefinitionError
from gui_detector_api.services.embeddings import CLIPEmbeddingService
from gui_detector_api.utils.images import LoadedImage, load_image_from_upload


@dataclass(slots=True)
class ClassExemplar:
    class_id: str
    class_name: str
    embedding: list[float]


class StoredUserClass(BaseModel):
    class_id: str
    name: str
    texts: list[str] = Field(default_factory=list)
    image_filenames: list[str] = Field(default_factory=list)
    text_embeddings: list[list[float]] = Field(default_factory=list)
    image_embeddings: list[list[float]] = Field(default_factory=list)


class StoredClassRegistry(BaseModel):
    classes: list[StoredUserClass] = Field(default_factory=list)


class ClassRegistryService:
    def __init__(self, settings, embedding_service: CLIPEmbeddingService) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.registry_dir = settings.class_registry_dir
        self.assets_dir = self.registry_dir / "assets"
        self.registry_path = self.registry_dir / "registry.json"
        self._lock = RLock()

        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._classes: dict[str, StoredUserClass] = {}
        self._load_from_disk()

    def list_classes(self) -> UserClassesResponse:
        with self._lock:
            classes = [self._to_response(record) for record in self._sorted_records()]
        return UserClassesResponse(classes=classes)

    def has_classes(self) -> bool:
        with self._lock:
            return bool(self._classes)

    def class_count(self) -> int:
        with self._lock:
            return len(self._classes)

    def get_exemplar_index(self) -> list[ClassExemplar]:
        with self._lock:
            exemplars: list[ClassExemplar] = []
            for record in self._classes.values():
                exemplars.extend(
                    ClassExemplar(class_id=record.class_id, class_name=record.name, embedding=embedding)
                    for embedding in record.text_embeddings
                )
                exemplars.extend(
                    ClassExemplar(class_id=record.class_id, class_name=record.name, embedding=embedding)
                    for embedding in record.image_embeddings
                )
            return exemplars

    async def create_class(
        self,
        *,
        name: str,
        texts: list[str] | None,
        images: list[UploadFile] | None,
    ) -> UserClassResponse:
        normalized_name = self._normalize_name(name)
        normalized_texts = self._normalize_texts(texts)
        loaded_images = await self._load_images(images)
        self._validate_exemplars(normalized_texts, loaded_images)

        text_embeddings = await self._embed_texts(normalized_texts)
        image_embeddings = await self._embed_images(loaded_images)
        class_id = uuid4().hex[:12]

        with self._lock:
            image_filenames = self._save_images(class_id, loaded_images)
            record = StoredUserClass(
                class_id=class_id,
                name=normalized_name,
                texts=normalized_texts,
                image_filenames=image_filenames,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            )
            self._classes[class_id] = record
            self._persist()
            return self._to_response(record)

    async def replace_class(
        self,
        *,
        class_id: str,
        name: str | None,
        texts: list[str] | None,
        images: list[UploadFile] | None,
    ) -> UserClassResponse:
        with self._lock:
            current = self._classes.get(class_id)
            if current is None:
                raise ClassNotFoundError(class_id)

        normalized_name = self._normalize_name(name) if name is not None else current.name
        normalized_texts = self._normalize_texts(texts)
        loaded_images = await self._load_images(images)
        self._validate_exemplars(normalized_texts, loaded_images)

        text_embeddings = await self._embed_texts(normalized_texts)
        image_embeddings = await self._embed_images(loaded_images)

        with self._lock:
            if class_id not in self._classes:
                raise ClassNotFoundError(class_id)
            image_filenames = self._save_images(class_id, loaded_images)
            record = StoredUserClass(
                class_id=class_id,
                name=normalized_name,
                texts=normalized_texts,
                image_filenames=image_filenames,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            )
            self._classes[class_id] = record
            self._persist()
            return self._to_response(record)

    def delete_class(self, class_id: str) -> None:
        with self._lock:
            record = self._classes.pop(class_id, None)
            if record is None:
                raise ClassNotFoundError(class_id)
            self._persist()

        asset_dir = self.assets_dir / class_id
        if asset_dir.exists():
            shutil.rmtree(asset_dir)

    def _load_from_disk(self) -> None:
        if not self.registry_path.exists():
            return

        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry = StoredClassRegistry.model_validate(payload)
        self._classes = {record.class_id: record for record in registry.classes}

    def _persist(self) -> None:
        payload = StoredClassRegistry(classes=list(self._classes.values())).model_dump(mode="json")
        temp_path = self.registry_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.registry_path)

    async def _load_images(self, images: list[UploadFile] | None) -> list[LoadedImage]:
        if not images:
            return []
        loaded_images: list[LoadedImage] = []
        for image in images:
            loaded_images.append(await load_image_from_upload(image, self.settings.max_upload_size_bytes))
        return loaded_images

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self.embedding_service.embed_texts, texts)

    async def _embed_images(self, images: list[LoadedImage]) -> list[list[float]]:
        if not images:
            return []
        pil_images = [loaded.image for loaded in images]
        return await asyncio.to_thread(self.embedding_service.embed_images, pil_images)

    def _save_images(self, class_id: str, images: list[LoadedImage]) -> list[str]:
        asset_dir = self.assets_dir / class_id
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
        if not images:
            return []
        asset_dir.mkdir(parents=True, exist_ok=True)

        filenames: list[str] = []
        for index, loaded in enumerate(images, start=1):
            filename = f"img-{index:03d}.png"
            loaded.image.save(asset_dir / filename, format="PNG")
            filenames.append(filename)
        return filenames

    def _sorted_records(self) -> list[StoredUserClass]:
        return sorted(self._classes.values(), key=lambda record: (record.name.lower(), record.class_id))

    def _to_response(self, record: StoredUserClass) -> UserClassResponse:
        return UserClassResponse(
            class_id=record.class_id,
            name=record.name,
            texts=record.texts,
            image_filenames=record.image_filenames,
            text_count=len(record.texts),
            image_count=len(record.image_filenames),
            exemplar_count=len(record.text_embeddings) + len(record.image_embeddings),
        )

    def _normalize_name(self, value: str | None) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise InvalidClassDefinitionError("Field 'name' is required.")
        return normalized

    def _normalize_texts(self, texts: list[str] | None) -> list[str]:
        return [value.strip() for value in texts or [] if value and value.strip()]

    def _validate_exemplars(self, texts: list[str], images: list[LoadedImage]) -> None:
        if not texts and not images:
            raise InvalidClassDefinitionError("Provide at least one text or image exemplar for a class.")
