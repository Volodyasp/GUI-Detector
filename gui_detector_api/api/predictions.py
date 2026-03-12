from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse

from gui_detector_api.domain.schemas import PredictionResponse
from gui_detector_api.errors import ModelUnavailableError
from gui_detector_api.rendering.preview import PreviewRenderer
from gui_detector_api.services.prediction import PredictionService

router = APIRouter(prefix="/v1", tags=["predictions"])


def get_prediction_service(request: Request) -> PredictionService:
    runtime = request.app.state.runtime
    return PredictionService(settings=runtime.settings, detector=runtime.detector)


def get_preview_renderer() -> PreviewRenderer:
    return PreviewRenderer()


@router.post("/predictions", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
):
    response, _ = await service.predict_upload(image)
    return response


@router.post("/predictions/preview", response_class=HTMLResponse)
async def preview_prediction(
    image: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
    renderer: PreviewRenderer = Depends(get_preview_renderer),
) -> HTMLResponse:
    if service.detector is None:
        raise ModelUnavailableError("The active detector is not ready.")

    response, source_image = await service.predict_upload(image)
    return HTMLResponse(renderer.render(response, source_image))
