from __future__ import annotations

from PIL import Image

from gui_detector_api.services.ocr import OcrBackend, OcrResult, OcrService


def test_none_backend_returns_empty():
    service = OcrService(backend=OcrBackend.NONE)
    result = service.extract_text(Image.new("RGB", (100, 30), "white"))
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.raw_texts == []


def test_preprocess_converts_to_grayscale():
    service = OcrService(backend=OcrBackend.NONE)
    rgb_image = Image.new("RGB", (200, 80), "red")
    processed = service._preprocess_for_ocr(rgb_image)
    assert processed.mode == "L"


def test_preprocess_upscales_small_crops():
    service = OcrService(backend=OcrBackend.NONE, min_crop_height=64)
    small_image = Image.new("RGB", (50, 20), "white")
    processed = service._preprocess_for_ocr(small_image)
    assert processed.height >= 64
    assert processed.width > 50


def test_preprocess_does_not_upscale_large_crops():
    service = OcrService(backend=OcrBackend.NONE, min_crop_height=64)
    large_image = Image.new("RGB", (200, 100), "white")
    processed = service._preprocess_for_ocr(large_image)
    assert processed.height == 100
    assert processed.width == 200


def test_easyocr_falls_back_to_none_when_not_installed(monkeypatch):
    service = OcrService(backend=OcrBackend.EASYOCR)
    monkeypatch.setattr("builtins.__import__", _block_import("easyocr"))
    service._ensure_loaded()
    assert service.backend == OcrBackend.NONE

    result = service.extract_text(Image.new("RGB", (100, 30), "white"))
    assert result.text == ""


def test_tesseract_falls_back_to_none_when_not_installed(monkeypatch):
    service = OcrService(backend=OcrBackend.TESSERACT)
    monkeypatch.setattr("builtins.__import__", _block_import("pytesseract"))
    service._ensure_loaded()
    assert service.backend == OcrBackend.NONE


def _block_import(blocked_module: str):
    original = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def guarded_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked: {blocked_module} not available")
        return original(name, *args, **kwargs)

    return guarded_import
