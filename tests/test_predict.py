"""Tests for the classifier wrapper in model/predict.py.

The real DistilBERT weights are replaced by a fake model that returns fixed
logits, so we test *our* logic (softmax, label mapping, safety escalation)
deterministically and without downloading anything.
"""
import types

import pytest

torch = pytest.importorskip("torch")
predict_module = pytest.importorskip("model.predict")

LABELS = predict_module.LABELS  # ["critical", "high", "medium", "low", "non_emergency"]


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[101, 102]])}


class FakeModel:
    def __init__(self, logits):
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, **inputs):
        return types.SimpleNamespace(logits=self._logits)


@pytest.fixture
def fake_classifier(monkeypatch):
    """Returns a function that installs a fake model producing the given logits."""

    def install(logits):
        monkeypatch.setattr(predict_module, "_load", lambda: None)
        monkeypatch.setattr(predict_module, "_tokenizer", FakeTokenizer())
        monkeypatch.setattr(predict_module, "_model", FakeModel(logits))

    return install


def test_confident_critical_prediction(fake_classifier):
    fake_classifier([6, 0, 0, 0, 0])
    result = predict_module.predict("he is not breathing")
    assert result["level"] == "critical"
    assert result["confidence"] > 0.9
    assert result["flagged"] is None


def test_scores_cover_every_label_and_sum_to_one(fake_classifier):
    fake_classifier([1, 2, 3, 2, 1])
    result = predict_module.predict("my son might have a broken wrist")
    assert list(result["all_scores"]) == LABELS
    assert sum(result["all_scores"].values()) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("logits, label", [
    ([0, 0, 0, 1, 0], "low"),            # ~40% confident
    ([0, 0, 0, 0, 1], "non_emergency"),  # ~40% confident
])
def test_low_confidence_low_severity_is_escalated_to_medium(fake_classifier, logits, label):
    fake_classifier(logits)
    result = predict_module.predict("i think it is nothing serious")
    assert result["level"] == "medium"
    assert f"escalated from '{label}'" in result["flagged"]


@pytest.mark.parametrize("logits, label", [
    ([0, 0, 0, 6, 0], "low"),
    ([0, 0, 0, 0, 6], "non_emergency"),
])
def test_confident_low_severity_is_not_escalated(fake_classifier, logits, label):
    fake_classifier(logits)
    result = predict_module.predict("i want to report a noise complaint")
    assert result["level"] == label
    assert result["flagged"] is None


def test_uncertain_critical_prediction_is_never_downgraded(fake_classifier):
    # Uniform logits: argmax picks index 0 (critical) at only 20% confidence.
    # The safety rule only escalates *up*, so this must stay critical.
    fake_classifier([0, 0, 0, 0, 0])
    result = predict_module.predict("help")
    assert result["level"] == "critical"
    assert result["flagged"] is None


@pytest.mark.parametrize("text", ["", "  ", "a", None])
def test_too_short_transcript_returns_unknown(fake_classifier, text):
    # NOTE: predict() currently calls _load() before this check, so the fixture also keeps
    # the test from touching the network. Moving the check above _load() would avoid
    # loading the model for empty input.
    fake_classifier([1, 1, 1, 1, 1])
    result = predict_module.predict(text)
    assert result["level"] == "unknown"
    assert result["confidence"] == 0.0
    assert "too short" in result["flagged"]
