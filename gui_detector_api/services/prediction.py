from __future__ import annotations

import asyncio
import logging

from fastapi import UploadFile

from gui_detector_api.detectors.base import Detector, DetectorError, DetectorPredictionError
from gui_detector_api.domain.schemas import ImageMetadata, PredictionResponse
from gui_detector_api.errors import ModelUnavailableError, PredictionExecutionError, PredictionTimeoutError
from gui_detector_api.settings import AppSettings
from gui_detector_api.utils.images import load_image_from_upload

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    async def predict_upload(self, detector: Detector | None, upload: UploadFile) -> PredictionResponse:
        if detector is None:
            raise ModelUnavailableError("The active detector is not ready.")

        loaded = await load_image_from_upload(upload, self.settings.max_upload_size_bytes)

        try:
            prediction = await asyncio.wait_for(
                asyncio.to_thread(detector.predict, loaded.image),
                timeout=self.settings.prediction_timeout_seconds,
            )
        except TimeoutError as exc:
            raise PredictionTimeoutError("Prediction exceeded the configured timeout.") from exc
        except DetectorPredictionError as exc:
            logger.exception("Prediction failed for active model '%s'.", self.settings.active_model)
            raise PredictionExecutionError(str(exc)) from exc
        except DetectorError as exc:
            logger.exception("Detector runtime error for active model '%s'.", self.settings.active_model)
            raise PredictionExecutionError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected prediction error for active model '%s'.", self.settings.active_model)
            raise PredictionExecutionError(str(exc)) from exc

        response = PredictionResponse(
            model=detector.info,
            image=ImageMetadata(
                filename=loaded.filename,
                content_type=loaded.content_type,
                width=loaded.image.width,
                height=loaded.image.height,
                size_bytes=loaded.size_bytes,
            ),
            detections=prediction.detections,
        )
        return response
