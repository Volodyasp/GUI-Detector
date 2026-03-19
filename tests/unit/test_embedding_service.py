from __future__ import annotations

from gui_detector_api.services.embeddings import EmbeddingService
from gui_detector_api.settings import AppSettings, default_models


class FakeBatchEncoding(dict):
    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    load_kwargs = None
    last_call_kwargs = None

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        cls.load_kwargs = {"repo_id": repo_id, **kwargs}
        return cls()

    def __call__(self, **kwargs):
        type(self).last_call_kwargs = kwargs
        return FakeBatchEncoding(pixel_values="pixels")


class FakeTensor:
    def __init__(self, rows) -> None:
        self.rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class FakeModel:
    load_kwargs = None

    def __init__(self) -> None:
        self.moved_to = None
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        cls.load_kwargs = {"repo_id": repo_id, **kwargs}
        return cls()

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True

    def get_text_features(self, **kwargs):
        del kwargs
        return FakeTensor([[3.0, 4.0]])

    def get_image_features(self, **kwargs):
        del kwargs
        return FakeTensor([[0.0, 5.0]])


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class FakeTorch:
    @staticmethod
    def no_grad():
        return FakeNoGrad()


def test_embedding_service_loads_runtime_and_normalizes_text_embeddings(monkeypatch, tmp_path):
    settings = AppSettings(
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    settings.embedding_model.device = "cpu"
    service = EmbeddingService(settings=settings)
    monkeypatch.setattr(service, "_import_runtime", lambda: (FakeModel, FakeProcessor, FakeTorch))

    embeddings = service.embed_texts(["primary button"])

    assert FakeProcessor.load_kwargs["repo_id"] == settings.embedding_model.hf_repo_id
    assert FakeModel.load_kwargs["repo_id"] == settings.embedding_model.hf_repo_id
    assert embeddings == [[0.6, 0.8]]
    assert service._model.moved_to == "cpu"
    assert service._model.eval_called is True


def test_embedding_service_normalizes_image_embeddings(monkeypatch, tmp_path):
    settings = AppSettings(
        models=default_models(),
        model_cache_dir=tmp_path / "model-cache",
        class_registry_dir=tmp_path / "class-registry",
    )
    settings.embedding_model.device = "cpu"
    service = EmbeddingService(settings=settings)
    monkeypatch.setattr(service, "_import_runtime", lambda: (FakeModel, FakeProcessor, FakeTorch))
    sentinel = object()

    embeddings = service.embed_images([sentinel])

    assert FakeProcessor.last_call_kwargs["images"] == [sentinel]
    assert embeddings == [[0.0, 1.0]]
