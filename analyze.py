# analyze.py
from model.predict import predict
from nlp_features import extract_features

def analyze_severity(transcript: str) -> dict:
    features = extract_features(transcript)
    result   = predict(transcript)
    result["nlp_signals"] = {
        "critical_keywords": features.get("critical_keywords", []),
        "urgency_words":     features.get("urgency_words",     []),
        "negations":         features.get("negated_words",     []),
    }
    return result