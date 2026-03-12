from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gui_detector_api.domain.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthcheck(request: Request) -> HealthResponse:
    runtime = request.app.state.runtime
    return HealthResponse(service=runtime.settings.service_name, version=runtime.settings.app_version)


@router.get("/readyz", response_model=ReadyResponse)
async def readiness(request: Request):
    runtime = request.app.state.runtime
    model_settings = runtime.settings.models.get(runtime.settings.active_model)
    if runtime.ready and runtime.detector is not None and model_settings is not None:
        return ReadyResponse(
            status="ready",
            active_model=runtime.settings.active_model,
            backend=model_settings.backend,
            detail="Active detector is loaded.",
        )

    return JSONResponse(
        status_code=503,
        content=ReadyResponse(
            status="not_ready",
            active_model=runtime.settings.active_model,
            backend=model_settings.backend if model_settings else None,
            detail=runtime.load_error or "Active detector has not been loaded.",
        ).model_dump(mode="json"),
    )
