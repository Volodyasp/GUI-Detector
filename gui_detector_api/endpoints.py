from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from gui_detector_api.domain.schemas import HealthResponse, PredictionResponse, ReadyResponse
from gui_detector_api.runtime import RuntimeState

router = APIRouter()


def get_runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime


def _build_ready_response(runtime: RuntimeState) -> ReadyResponse:
    model_settings = runtime.settings.models.get(runtime.settings.active_model)
    return ReadyResponse(
        status="ready",
        active_model=runtime.settings.active_model,
        backend=model_settings.backend if model_settings is not None else None,
        detail="Active detector is loaded.",
    )


def _build_not_ready_response(runtime: RuntimeState) -> JSONResponse:
    model_settings = runtime.settings.models.get(runtime.settings.active_model)
    payload = ReadyResponse(
        status="not_ready",
        active_model=runtime.settings.active_model,
        backend=model_settings.backend if model_settings else None,
        detail=runtime.load_error or "Active detector has not been loaded.",
    )
    return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))


async def _run_prediction(runtime: RuntimeState, image: UploadFile) -> tuple[PredictionResponse, object]:
    return await runtime.prediction_service.predict_upload(runtime.detector, image)


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthcheck(runtime: RuntimeState = Depends(get_runtime)) -> HealthResponse:
    return HealthResponse(service=runtime.settings.service_name, version=runtime.settings.app_version)


@router.get("/readyz", response_model=ReadyResponse, tags=["health"])
async def readiness(runtime: RuntimeState = Depends(get_runtime)):
    model_settings = runtime.settings.models.get(runtime.settings.active_model)
    if runtime.ready and runtime.detector is not None and model_settings is not None:
        return _build_ready_response(runtime)
    return _build_not_ready_response(runtime)


@router.post("/v1/predictions", response_model=PredictionResponse, tags=["predictions"])
async def predict(
    image: UploadFile = File(...),
    runtime: RuntimeState = Depends(get_runtime),
):
    response, _ = await _run_prediction(runtime, image)
    return response


@router.post("/v1/predictions/preview", response_class=HTMLResponse, tags=["predictions"])
async def preview_prediction(
    image: UploadFile = File(...),
    runtime: RuntimeState = Depends(get_runtime),
) -> HTMLResponse:
    response, source_image = await _run_prediction(runtime, image)
    return HTMLResponse(runtime.preview_renderer.render(response, source_image))


def register_endpoints(app: FastAPI) -> None:
    app.include_router(router)
