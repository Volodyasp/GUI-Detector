from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from gui_detector_api.domain.schemas import (
    HealthResponse,
    PredictionResponse,
    ReadyResponse,
    UserClassResponse,
    UserClassesResponse,
)
from gui_detector_api.errors import APIError
from gui_detector_api.rendering.app_page import render_app_page
from gui_detector_api.runtime import RuntimeState

page_router = APIRouter()
api_router = APIRouter(prefix="/v1")


def get_runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime


def _reject_legacy_prediction_query_params(request: Request) -> None:
    if "image_format" in request.query_params:
        raise APIError(
            status_code=400,
            error="unsupported_parameter",
            detail="Query parameter 'image_format' is no longer supported. Use the web UI at '/' or render bounding boxes on the client.",
        )
    if "include_image" in request.query_params:
        raise APIError(
            status_code=400,
            error="unsupported_parameter",
            detail="Query parameter 'include_image' is no longer supported. Use the web UI at '/' or render bounding boxes on the client.",
        )


async def _reject_legacy_prediction_form_fields(request: Request) -> None:
    form = await request.form()
    legacy_fields = [name for name in ("query_texts", "query_image") if name in form]
    if not legacy_fields:
        return
    raise APIError(
        status_code=400,
        error="unsupported_parameter",
        detail=(
            "Fields 'query_texts' and 'query_image' are no longer supported on /v1/predictions. "
            "Create server-side classes via /v1/classes and use CLIP post-classification instead."
        ),
    )


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


@page_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def app_shell() -> HTMLResponse:
    return HTMLResponse(render_app_page())


@api_router.get("/healthcheck", response_model=HealthResponse, tags=["health"])
async def healthcheck(runtime: RuntimeState = Depends(get_runtime)) -> HealthResponse:
    return HealthResponse(service=runtime.settings.service_name, version=runtime.settings.app_version)


@api_router.get("/readiness", response_model=ReadyResponse, tags=["health"])
async def readiness(runtime: RuntimeState = Depends(get_runtime)):
    if runtime.ready and runtime.detector is not None:
        return _build_ready_response(runtime)
    return _build_not_ready_response(runtime)


@api_router.get("/classes", response_model=UserClassesResponse, tags=["classes"])
async def list_classes(runtime: RuntimeState = Depends(get_runtime)) -> UserClassesResponse:
    return runtime.class_registry.list_classes()


@api_router.post("/classes", response_model=UserClassResponse, tags=["classes"])
async def create_class(
    name: Annotated[str, Form(...)],
    texts: Annotated[list[str] | None, Form()] = None,
    images: Annotated[list[UploadFile] | None, File()] = None,
    runtime: RuntimeState = Depends(get_runtime),
) -> UserClassResponse:
    return await runtime.class_registry.create_class(
        name=name,
        texts=texts,
        images=images,
    )


@api_router.put("/classes/{class_id}", response_model=UserClassResponse, tags=["classes"])
async def replace_class(
    class_id: str,
    name: Annotated[str | None, Form()] = None,
    texts: Annotated[list[str] | None, Form()] = None,
    images: Annotated[list[UploadFile] | None, File()] = None,
    runtime: RuntimeState = Depends(get_runtime),
) -> UserClassResponse:
    return await runtime.class_registry.replace_class(
        class_id=class_id,
        name=name,
        texts=texts,
        images=images,
    )


@api_router.delete("/classes/{class_id}", status_code=204, tags=["classes"])
async def delete_class(class_id: str, runtime: RuntimeState = Depends(get_runtime)) -> Response:
    runtime.class_registry.delete_class(class_id)
    return Response(status_code=204)


@api_router.post(
    "/predictions",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["predictions"],
)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    runtime: RuntimeState = Depends(get_runtime),
) -> PredictionResponse:
    _reject_legacy_prediction_query_params(request)
    await _reject_legacy_prediction_form_fields(request)
    return await runtime.prediction_service.predict_upload(runtime.detector, image)


def register_endpoints(app: FastAPI) -> None:
    app.include_router(page_router)
    app.include_router(api_router)
