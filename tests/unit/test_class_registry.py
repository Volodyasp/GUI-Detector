from __future__ import annotations

import asyncio
from io import BytesIO

from starlette.datastructures import UploadFile

from gui_detector_api.services.class_registry import ClassRegistryService
from gui_detector_api.settings import AppSettings, default_models


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_images(self, images):
        return [[0.0, 1.0] for _ in images]


def make_upload(filename: str, payload: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(payload), headers={"content-type": "image/png"})


def test_class_registry_persists_and_cleans_assets(tmp_path, png_bytes):
    settings = AppSettings(
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    service = ClassRegistryService(settings=settings, embedding_service=FakeEmbeddingService())

    created = asyncio.run(
        service.create_class(
            name="primary_button",
            texts=["button", "cta"],
            images=[make_upload("reference.png", png_bytes)],
        )
    )

    assert created.name == "primary_button"
    assert created.exemplar_count == 3
    asset_dir = settings.class_registry_dir / "assets" / created.class_id
    assert asset_dir.exists()
    assert list(asset_dir.iterdir())

    reloaded = ClassRegistryService(settings=settings, embedding_service=FakeEmbeddingService())
    listed = reloaded.list_classes()
    assert listed.classes[0].class_id == created.class_id

    replaced = asyncio.run(
        reloaded.replace_class(
            class_id=created.class_id,
            name="secondary_button",
            texts=["secondary action"],
            images=[make_upload("replacement.png", png_bytes)],
        )
    )
    assert replaced.name == "secondary_button"
    assert replaced.text_count == 1
    assert replaced.image_count == 1

    reloaded.delete_class(created.class_id)
    assert reloaded.list_classes().classes == []
    assert not asset_dir.exists()
