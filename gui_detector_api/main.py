from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from gui_detector_api.detectors.model_registry import ModelRegistry
from gui_detector_api.endpoints import register_endpoints
from gui_detector_api.errors import APIError, api_error_handler, unexpected_error_handler, validation_error_handler
from gui_detector_api.lifespan import create_lifespan
from gui_detector_api.settings import AppSettings, get_settings


def create_app(
    *,
    settings: AppSettings | None = None,
    model_registry: ModelRegistry | None = None,
    load_detector_on_startup: bool = True,
) -> FastAPI:
    app_settings = settings or get_settings()
    resolved_registry = model_registry or ModelRegistry()

    app = FastAPI(
        title=app_settings.service_name,
        version=app_settings.app_version,
        lifespan=create_lifespan(
            settings=app_settings,
            model_registry=resolved_registry,
            load_detector_on_startup=load_detector_on_startup,
        ),
    )
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    register_endpoints(app)
    return app


app = create_app()
