FROM python:3.12-slim

ENV POETRY_VERSION=2.1.4 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
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
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()"

CMD ["uvicorn", "gui_detector_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
