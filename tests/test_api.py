"""Tests for the Flask API layer (app.py)."""
import io

from tests.conftest import FAKE_SEVERITY, FAKE_TRANSCRIPT


def test_health_check(api):
    response = api.client.get("/")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_analyze_returns_transcript_and_severity(api):
    response = api.post_audio()
    assert response.status_code == 200
    body = response.get_json()
    assert body["transcript"] == FAKE_TRANSCRIPT
    assert body["severity"]["level"] == FAKE_SEVERITY["level"]
    assert set(body["severity"]) >= {"level", "confidence", "all_scores", "flagged", "nlp_signals"}


def test_missing_audio_field_returns_400(api):
    response = api.post_audio(field="wrong_field")
    assert response.status_code == 400
    assert response.get_json() == {"error": "No audio file provided"}


def test_empty_audio_file_returns_400(api):
    response = api.post_audio(data=b"")
    assert response.status_code == 400
    assert response.get_json() == {"error": "Empty audio file"}


def test_no_speech_detected_returns_400(api, monkeypatch):
    monkeypatch.setattr(api.module, "transcribe_audio", lambda b: {"transcript": "   "})
    response = api.post_audio()
    assert response.status_code == 400
    assert response.get_json() == {"error": "No speech detected"}


def test_internal_errors_return_500_without_leaking_details(api, monkeypatch):
    def boom(_audio_bytes):
        raise RuntimeError("secret internal path /opt/models/xyz")

    monkeypatch.setattr(api.module, "transcribe_audio", boom)
    response = api.post_audio()
    assert response.status_code == 500
    assert response.get_json() == {"error": "Internal processing error"}
    assert "secret" not in response.get_data(as_text=True)


def test_upload_over_size_limit_is_rejected(api):
    api.module.app.config["MAX_CONTENT_LENGTH"] = 1024
    response = api.client.post(
        "/analyze",
        data={"audio": (io.BytesIO(b"x" * 4096), "big.webm")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413


def test_analyze_rejects_get_requests(api):
    assert api.client.get("/analyze").status_code == 405
