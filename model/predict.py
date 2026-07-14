# model/predict.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS    = ["critical", "high", "medium", "low", "non_emergency"]
SAVE_PATH = "chirantana20/emergency-severity-distilbert"  # HF Hub repo ID
MAX_LEN   = 128

# ── Load once at startup ──────────────────────────────────────────────────────
# These are module-level variables — loaded once when Flask starts,
# not on every request. Reloading a 268MB model per request would be ~2s lag.
_tokenizer = None
_model     = None

def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        print("Loading fine-tuned model...")
        _tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
        _model     = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)
        _model.eval()  # disable dropout for deterministic inference
        print("Model ready.")


def predict(transcript: str) -> dict:
    """
    Takes a raw transcript string.
    Returns severity level, confidence, and all class scores.
    """
    _load()

    # Edge case — empty or very short transcript
    if not transcript or len(transcript.strip()) < 3:
        return {
            "level":      "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "flagged":    "transcript too short to classify"
        }

    # Tokenize
    inputs = _tokenizer(
        transcript,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    # Forward pass — no_grad saves memory, not computing gradients at inference
    with torch.no_grad():
        logits = _model(**inputs).logits

    # Convert logits → probabilities via softmax
    # Logits are raw scores — softmax normalizes them to sum to 1.0
    probs      = torch.softmax(logits, dim=-1)[0]
    pred_id    = torch.argmax(probs).item()
    confidence = probs[pred_id].item()
    level      = LABELS[pred_id]

    # All class scores — useful for showing a probability bar in the UI
    all_scores = {
        label: round(probs[i].item(), 3)
        for i, label in enumerate(LABELS)
    }

    result = {
        "level":      level,
        "confidence": round(confidence, 3),
        "all_scores": all_scores,
        "flagged":    None
    }

    # ── Safety rule ───────────────────────────────────────────────────────────
    # Based on the confusion matrix: model under-predicts when uncertain.
    # If it says low/non_emergency but isn't confident, escalate to medium.
    # Better a false alarm than a missed critical call.
    if level in ["low", "non_emergency"] and confidence < 0.80:
        result["level"]   = "medium"
        result["flagged"] = f"escalated from '{level}' — confidence {confidence:.0%} below threshold"

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("he's not breathing i can't find a pulse",          "critical"),
        ("there's a fire in the building people are inside", "high"),
        ("my son fell and hurt his wrist might be broken",   "medium"),
        ("i have a minor cut on my finger",                  "low"),
        ("i want to report a noise complaint",               "non_emergency"),
        # Tricky cases — negation + mixed signals
        ("he says he's fine but he can't move his legs",     "critical"),
        ("she's not in pain but she fainted twice today",    "high"),
    ]

    print(f"{'Transcript':<45} {'Expected':<15} {'Predicted':<15} {'Conf':>6}  {'Flag'}")
    print("─" * 110)

    for transcript, expected in tests:
        result = predict(transcript)
        match  = "✓" if result["level"] == expected else "✗"
        flag   = result["flagged"] or ""
        print(
            f"{transcript[:44]:<45} "
            f"{expected:<15} "
            f"{result['level']:<15} "
            f"{result['confidence']:>5.0%}  "
            f"{match}  {flag}"
        )