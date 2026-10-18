"""Tests for analyze.py, which merges classifier output with NLP signals."""
import pytest

try:
    import analyze
except (ImportError, OSError):  # torch / spaCy / model not available
    pytest.skip("ML dependencies not available", allow_module_level=True)


def test_analyze_severity_merges_prediction_and_nlp_signals(monkeypatch):
    monkeypatch.setattr(analyze, "predict", lambda text: {"level": "critical", "confidence": 0.95})
    monkeypatch.setattr(
        analyze,
        "extract_features",
        lambda text: {
            "critical_keywords": ["not breathing"],
            "urgency_words": ["hurry"],
            "negated_words": ["breathing"],
        },
    )

    result = analyze.analyze_severity("he is not breathing please hurry")

    assert result["level"] == "critical"
    assert result["confidence"] == 0.95
    assert result["nlp_signals"] == {
        "critical_keywords": ["not breathing"],
        "urgency_words": ["hurry"],
        "negations": ["breathing"],  # renamed from negated_words in the public output
    }


def test_analyze_severity_handles_missing_features(monkeypatch):
    monkeypatch.setattr(analyze, "predict", lambda text: {"level": "low"})
    monkeypatch.setattr(analyze, "extract_features", lambda text: {})

    result = analyze.analyze_severity("minor cut")

    assert result["nlp_signals"] == {"critical_keywords": [], "urgency_words": [], "negations": []}
