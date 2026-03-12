from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from gui_detector_api.api.health import router as health_router
from gui_detector_api.api.predictions import router as predictions_router
from gui_detector_api.detectors.factory import DetectorFactory
from gui_detector_api.errors import APIError, api_error_handler, unexpected_error_handler, validation_error_handler
from gui_detector_api.runtime import RuntimeState
from gui_detector_api.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: AppSettings | None = None,
    detector_factory: DetectorFactory | None = None,
    load_detector_on_startup: bool = True,
) -> FastAPI:
    app_settings = settings or get_settings()
    factory = detector_factory or DetectorFactory()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=getattr(logging, app_settings.log_level.upper(), logging.INFO))
        runtime = RuntimeState(settings=app_settings, detector_factory=factory)
        app.state.runtime = runtime

        if load_detector_on_startup:
            try:
                detector = factory.create(app_settings)
                await asyncio.to_thread(detector.load)
                runtime.mark_ready(detector)
                logger.info("Loaded active detector '%s'.", app_settings.active_model)
            except Exception as exc:
                logger.exception("Failed to load active detector '%s'.", app_settings.active_model)
                runtime.mark_failed(str(exc))

        yield

    app = FastAPI(
        title=app_settings.service_name,
        version=app_settings.app_version,
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    app.include_router(health_router)
    app.include_router(predictions_router)
    return app


app = create_app()
