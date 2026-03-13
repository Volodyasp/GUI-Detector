from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gui_detector_api.domain.schemas import ErrorResponse


class APIError(Exception):
    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail


class InvalidUploadError(APIError):
    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(status_code=status_code, error="invalid_upload", detail=detail)


class PayloadTooLargeError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=413, error="payload_too_large", detail=detail)


class ModelUnavailableError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=503, error="model_unavailable", detail=detail)


class EmbeddingUnavailableError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=503, error="embedding_unavailable", detail=detail)


class PredictionTimeoutError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=504, error="prediction_timeout", detail=detail)


class PredictionExecutionError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=500, error="prediction_failed", detail=detail)


class ClassificationExecutionError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=500, error="classification_failed", detail=detail)


class InvalidClassDefinitionError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=400, error="invalid_class_definition", detail=detail)


class ClassNotFoundError(APIError):
    def __init__(self, class_id: str) -> None:
        super().__init__(status_code=404, error="class_not_found", detail=f"Class '{class_id}' does not exist.")


async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.error, detail=exc.detail).model_dump(),
    )


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=str(exc)).model_dump(),
    )


async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="internal_error", detail=str(exc)).model_dump(),
    )
