from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class OcrBackend(StrEnum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    NONE = "none"


@dataclass(slots=True)
class OcrResult:
    text: str
    confidence: float
    raw_texts: list[str]


class OcrService:
    def __init__(self, backend: OcrBackend, *, languages: list[str] | None = None, min_crop_height: int = 64) -> None:
        self.backend = backend
        self.languages = languages or ["en"]
        self.min_crop_height = min_crop_height
        self._engine = None
        self._lock = RLock()

    def extract_text(self, image: Image.Image) -> OcrResult:
        if self.backend == OcrBackend.NONE:
            return OcrResult(text="", confidence=0.0, raw_texts=[])
        self._ensure_loaded()
        preprocessed = self._preprocess_for_ocr(image)
        return self._run_ocr(preprocessed)

    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        from PIL import Image as PILImage

        if image.mode != "L":
            image = image.convert("L")
        if image.height < self.min_crop_height:
            scale = max(2, self.min_crop_height // max(1, image.height) + 1)
            image = image.resize(
                (image.width * scale, image.height * scale),
                PILImage.LANCZOS,
            )
        return image

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._engine is not None:
                return
            try:
                if self.backend == OcrBackend.EASYOCR:
                    self._engine = self._load_easyocr()
                elif self.backend == OcrBackend.TESSERACT:
                    self._engine = self._load_tesseract()
                elif self.backend == OcrBackend.PADDLEOCR:
                    self._engine = self._load_paddleocr()
            except RuntimeError as exc:
                logger.warning(
                    "OCR backend '%s' unavailable, falling back to visual-only classification: %s",
                    self.backend,
                    exc,
                )
                self.backend = OcrBackend.NONE

    def _run_ocr(self, image: Image.Image) -> OcrResult:
        with self._lock:
            if self.backend == OcrBackend.EASYOCR:
                return self._run_easyocr(image)
            elif self.backend == OcrBackend.TESSERACT:
                return self._run_tesseract(image)
            elif self.backend == OcrBackend.PADDLEOCR:
                return self._run_paddleocr(image)
        return OcrResult(text="", confidence=0.0, raw_texts=[])

    def _load_easyocr(self):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed. Run `poetry install --with ocr` to enable it."
            ) from exc
        return easyocr.Reader(self.languages, gpu=False)

    def _run_easyocr(self, image: Image.Image) -> OcrResult:
        import numpy as np

        results = self._engine.readtext(np.array(image))
        if not results:
            return OcrResult(text="", confidence=0.0, raw_texts=[])
        raw_texts = [text for _, text, _ in results]
        confidences = [conf for _, _, conf in results]
        joined = " ".join(raw_texts).strip().lower()
        avg_conf = sum(confidences) / len(confidences)
        return OcrResult(text=joined, confidence=avg_conf, raw_texts=raw_texts)

    def _load_tesseract(self):
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is not installed. Run `pip install pytesseract` and install the Tesseract binary."
            ) from exc
        return pytesseract

    def _run_tesseract(self, image: Image.Image) -> OcrResult:
        data = self._engine.image_to_data(image, output_type=self._engine.Output.DICT)
        raw_texts = []
        confidences = []
        for text, conf in zip(data["text"], data["conf"]):
            if int(conf) > 0 and text.strip():
                raw_texts.append(text.strip())
                confidences.append(int(conf) / 100.0)
        if not raw_texts:
            return OcrResult(text="", confidence=0.0, raw_texts=[])
        joined = " ".join(raw_texts).strip().lower()
        avg_conf = sum(confidences) / len(confidences)
        return OcrResult(text=joined, confidence=avg_conf, raw_texts=raw_texts)

    def _load_paddleocr(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Run `pip install paddleocr paddlepaddle` to enable it."
            ) from exc
        return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def _run_paddleocr(self, image: Image.Image) -> OcrResult:
        import numpy as np

        result = self._engine.ocr(np.array(image), cls=True)
        if not result or not result[0]:
            return OcrResult(text="", confidence=0.0, raw_texts=[])
        raw_texts = [line[1][0] for line in result[0]]
        confidences = [line[1][1] for line in result[0]]
        joined = " ".join(raw_texts).strip().lower()
        avg_conf = sum(confidences) / len(confidences)
        return OcrResult(text=joined, confidence=avg_conf, raw_texts=raw_texts)
