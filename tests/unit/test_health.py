from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthcheck_returns_service_metadata(ready_app):
    with TestClient(ready_app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "gui-detector-api"


def test_readiness_returns_503_when_detector_is_not_loaded(not_ready_app):
    with TestClient(not_ready_app) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readiness_returns_200_when_detector_is_loaded(ready_app):
    with TestClient(ready_app) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_app_starts_and_stops_cleanly(ready_app):
    with TestClient(ready_app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
