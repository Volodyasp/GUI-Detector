from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_ui_returns_html_shell(ready_app):
    with TestClient(ready_app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "GUI Detector Studio" in response.text
    assert 'id="image-input"' in response.text
    assert 'id="detect-button"' in response.text
    assert 'id="download-button"' in response.text
    assert 'id="preview-canvas"' in response.text
    assert 'fetch("/v1/readiness")' in response.text
    assert 'fetch("/v1/predictions"' in response.text
    assert 'query-texts' not in response.text
    assert 'query-image-input' not in response.text


def test_removed_preview_endpoint_returns_404(ready_app, png_bytes):
    with TestClient(ready_app) as client:
        response = client.post(
            "/v1/predictions/preview",
            files={"image": ("sample.png", png_bytes, "image/png")},
        )

    assert response.status_code == 404
