"""Shared fixtures.

The `api` fixture imports app.py with `transcribe` and `analyze` replaced by
lightweight stubs. That keeps the API tests fast and independent of Whisper,
PyTorch and the Hugging Face Hub, so they exercise only the Flask layer:
validation, status codes and error handling.
"""
import io
import sys
import types
from dataclasses import dataclass

import pytest

FAKE_TRANSCRIPT = "he is not breathing please hurry"
FAKE_SEVERITY = {
    "level": "critical",
    "confidence": 0.97,
    "all_scores": {
        "critical": 0.97,
        "high": 0.02,
        "medium": 0.01,
        "low": 0.0,
        "non_emergency": 0.0,
    },
    "flagged": None,
    "nlp_signals": {
        "critical_keywords": ["not breathing"],
        "urgency_words": ["hurry"],
        "negations": [],
    },
}


@dataclass
class Api:
    client: object
    module: types.ModuleType

    def post_audio(self, data: bytes = b"fake-audio-bytes", field: str = "audio"):
        return self.client.post(
            "/analyze",
            data={field: (io.BytesIO(data), "call.webm")},
            content_type="multipart/form-data",
        )


@pytest.fixture
def api(monkeypatch):
    fake_transcribe = types.ModuleType("transcribe")
    fake_transcribe.transcribe_audio = lambda audio_bytes: {"transcript": FAKE_TRANSCRIPT}
    fake_analyze = types.ModuleType("analyze")
    fake_analyze.analyze_severity = lambda transcript: dict(FAKE_SEVERITY)

    monkeypatch.setitem(sys.modules, "transcribe", fake_transcribe)
    monkeypatch.setitem(sys.modules, "analyze", fake_analyze)
    sys.modules.pop("app", None)

    import app as app_module

    app_module.app.config["TESTING"] = True
    yield Api(client=app_module.app.test_client(), module=app_module)

    # Don't leak an `app` module bound to the stubs into other tests
    sys.modules.pop("app", None)
