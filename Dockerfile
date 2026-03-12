FROM python:3.12-slim

ENV POETRY_VERSION=2.1.4 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY poetry.lock pyproject.toml README.md ./
COPY gui_detector_api ./gui_detector_api

RUN poetry install --without dev --with models --no-interaction --no-ansi

RUN useradd --create-home --shell /bin/bash appuser
RUN mkdir -p /app/model-cache && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/readiness').getcode() == 200 else 1)"

CMD ["uvicorn", "gui_detector_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
