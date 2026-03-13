from __future__ import annotations

import json

from gui_detector_api.settings import AppSettings, get_settings


def test_settings_use_defaults():
    settings = AppSettings()
    assert settings.active_model == "ui_detr_1"
    assert "ui_detr_1" in settings.models
    assert "gpa_gui_detector" in settings.models
    assert settings.models["gpa_gui_detector"].device == "auto"
    assert settings.embedding_model.hf_repo_id == "openai/clip-vit-large-patch14"
    assert settings.classification_knn_k == 3


def test_settings_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("GUI_DETECTOR_ACTIVE_MODEL", "ui_detr_1")
    monkeypatch.setenv(
        "GUI_DETECTOR_MODELS",
        json.dumps(
            {
                "ui_detr_1": {
                    "backend": "rfdetr",
                    "hf_repo_id": "racineai/UI-DETR-1",
                    "weight_filename": "local-model.pth",
                    "device": "mps",
                    "confidence_threshold": 0.5,
                }
            }
        ),
    )
    monkeypatch.setenv("GUI_DETECTOR_EMBEDDING_MODEL__HF_REPO_ID", "openai/clip-vit-large-patch14")
    monkeypatch.setenv("GUI_DETECTOR_EMBEDDING_MODEL__DEVICE", "cpu")
    monkeypatch.setenv("GUI_DETECTOR_EMBEDDING_MODEL__LOCAL_FILES_ONLY", "true")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.active_model == "ui_detr_1"
    assert settings.models["ui_detr_1"].weight_filename == "local-model.pth"
    assert settings.models["ui_detr_1"].device == "mps"
    assert settings.embedding_model.device == "cpu"
    assert settings.embedding_model.local_files_only is True
    get_settings.cache_clear()
