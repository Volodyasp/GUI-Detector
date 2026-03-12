from __future__ import annotations

import json

from gui_detector_api.settings import AppSettings, get_settings


def test_settings_use_defaults():
    settings = AppSettings()
    assert settings.active_model == "gpa_gui_detector"
    assert "ui_detr_1" in settings.models
    assert settings.models["gpa_gui_detector"].device == "auto"


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
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.active_model == "ui_detr_1"
    assert settings.models["ui_detr_1"].weight_filename == "local-model.pth"
    assert settings.models["ui_detr_1"].device == "mps"
    get_settings.cache_clear()
