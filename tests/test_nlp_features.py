"""Tests for the spaCy feature extraction in nlp_features.py."""
import pytest

try:
    from nlp_features import extract_features
except (ImportError, OSError):  # spaCy or en_core_web_sm not installed
    pytest.skip("spaCy / en_core_web_sm not available", allow_module_level=True)


def test_detects_critical_keywords():
    features = extract_features("He is unconscious and there is chest pain")
    assert "unconscious" in features["critical_keywords"]
    assert "chest pain" in features["critical_keywords"]


def test_detects_high_keywords_and_urgency_words():
    features = extract_features("There was a car crash, please hurry")
    assert "crash" in features["high_keywords"]
    assert "please" in features["urgency_words"]
    assert "hurry" in features["urgency_words"]


def test_detects_negation():
    # "not breathing" and "breathing" mean opposite things; the head word is captured
    features = extract_features("He is not breathing")
    assert "breathing" in features["negated_words"]


def test_matching_is_case_insensitive():
    features = extract_features("HE IS NOT BREATHING")
    assert "not breathing" in features["critical_keywords"]


@pytest.mark.parametrize("text, expected_hint", [
    ("He is unconscious", "critical"),
    ("There is a fire", "high"),
    ("Please help me", "medium"),
    ("I would like to report a noise complaint", "unknown"),
])
def test_keyword_severity_hint_priority(text, expected_hint):
    assert extract_features(text)["keyword_severity_hint"] == expected_hint


def test_critical_takes_priority_over_high():
    features = extract_features("There is a fire and he is unconscious")
    assert features["keyword_severity_hint"] == "critical"


def test_sentence_statistics():
    features = extract_features("He fell. He is bleeding. Please come.")
    assert features["sentence_count"] == 3
    assert features["avg_sentence_length"] > 0


def test_empty_input_does_not_crash():
    features = extract_features("")
    assert features["critical_keywords"] == []
    assert features["sentence_count"] == 0
    assert features["avg_sentence_length"] == 0
    assert features["verb_ratio"] == 0


@pytest.mark.xfail(
    strict=True,
    reason="Known bug: keywords use substring matching, so 'now' matches 'know' and 'dead' "
    "matches 'deadline'. Fix with word-boundary matching, then remove this marker.",
)
def test_keywords_should_match_whole_words_only():
    features = extract_features("I know the deadline is tomorrow")
    assert features["urgency_words"] == []
