from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_web_panel_and_static_assets() -> None:
    page = client.get("/")
    styles = client.get("/static/styles.css")
    script = client.get("/static/app.js")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "CHARACTER.EXTRACTOR" in page.text
    assert 'id="progress-track"' in page.text
    assert 'id="character-grid"' in page.text
    assert styles.status_code == 200
    assert "gradient" not in styles.text.lower()
    assert "body:not(.has-job)" in styles.text
    assert script.status_code == 200
    assert "/v1/jobs" in script.text
    assert 'classList.add("has-job")' in script.text


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["analysis_model"] == "gemini-3.1-flash-lite"
    assert payload["detection_mode"] in {"universal", "people"}
    assert payload["queue_mode"] == "serialized"
    assert payload["requested_device"] in {"auto", "cpu", "cuda", "mps"}
    assert isinstance(payload["weights_ready"], bool)


def test_rejects_non_mp4() -> None:
    from app import main

    original_key = main.settings.google_api_key
    object.__setattr__(main.settings, "google_api_key", "test-key")
    response = client.post(
        "/v1/jobs",
        files={"video": ("clip.mov", b"not a video", "video/quicktime")},
    )
    object.__setattr__(main.settings, "google_api_key", original_key)
    assert response.status_code == 415


def test_job_requires_api_key() -> None:
    from app import main

    original_key = main.settings.google_api_key
    object.__setattr__(main.settings, "google_api_key", None)
    response = client.post(
        "/v1/jobs",
        files={"video": ("clip.mp4", b"not a video", "video/mp4")},
    )
    object.__setattr__(main.settings, "google_api_key", original_key)
    assert response.status_code == 503
