from __future__ import annotations

from fastapi import APIRouter, FastAPI

from gui_detector_api.api.health import router as health_router
from gui_detector_api.api.predictions import router as predictions_router

router = APIRouter()
router.include_router(health_router)
router.include_router(predictions_router)


def register_endpoints(app: FastAPI) -> None:
    app.include_router(router)
