from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from gui_detector_api.detectors.model_registry import ModelRegistry
from gui_detector_api.runtime import RuntimeState
from gui_detector_api.settings import AppSettings

logger = logging.getLogger(__name__)


def create_lifespan(
    *,
    settings: AppSettings,
    model_registry: ModelRegistry,
    load_detector_on_startup: bool = True,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
        runtime = RuntimeState(settings=settings)
        app.state.runtime = runtime

        if load_detector_on_startup:
            detector = None
            try:
                detector = model_registry.create_active_detector(settings)
                await asyncio.to_thread(detector.load)
                runtime.mark_ready(detector)
                logger.info(
                    "Loaded active detector '%s' on device '%s'.",
                    settings.active_model,
                    detector.resolved_device or "unknown",
                )
            except Exception as exc:
                logger.exception(
                    "Failed to load active detector '%s' (requested device '%s', resolved device '%s').",
                    settings.active_model,
                    settings.models.get(settings.active_model).device if settings.active_model in settings.models else "unknown",
                    detector.resolved_device if detector is not None else "unresolved",
                )
                runtime.mark_failed(str(exc))

        yield

    return lifespan
