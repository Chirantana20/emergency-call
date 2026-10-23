import spacy

nlp = spacy.load("en_core_web_sm")

# Keywords that strongly signal each severity tier
CRITICAL_KEYWORDS = {
    "not breathing", "no pulse", "unconscious", "unresponsive",
    "heart attack", "stroke", "choking", "drowning", "shooting",
    "stabbing", "overdose", "seizure", "chest pain"
}

HIGH_KEYWORDS = {
    "bleeding", "broken", "fracture", "burned", "fire", "crash",
    "accident", "attack", "trapped", "severe", "can't breathe"
}

URGENCY_WORDS = {
    "help", "please", "hurry", "quick", "fast", "now",
    "dying", "dead", "emergency", "urgent"
}

def extract_features(transcript: str) -> dict:
    """
    Run spaCy NLP pipeline and extract features
    that help Claude make a better severity judgment.
    """
    doc = nlp(transcript.lower())

    # --- Named Entity Recognition ---
    # spaCy can find people, places, organizations in the text
    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]

    # --- Keyword matching (exact + partial) ---
    text_lower = transcript.lower()
    critical_hits = [kw for kw in CRITICAL_KEYWORDS if kw in text_lower]
    high_hits     = [kw for kw in HIGH_KEYWORDS     if kw in text_lower]
    urgency_hits  = [kw for kw in URGENCY_WORDS     if kw in text_lower]

    # --- Sentence-level stats ---
    sentences = list(doc.sents)
    avg_sentence_length = (
        sum(len(s) for s in sentences) / len(sentences)
        if sentences else 0
    )

    # --- Part-of-speech signals ---
    # High ratio of verbs often = action/event happening (vs describing)
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    token_count = len([t for t in doc if not t.is_punct])
    verb_ratio = verb_count / token_count if token_count > 0 else 0

    # --- Negation detection ---
    # "not breathing" is very different from "breathing"
    negations = [
        token.head.text
        for token in doc
        if token.dep_ == "neg"  # spaCy marks negation dependencies
    ]

    return {
        "entities":             entities,
        "critical_keywords":    critical_hits,
        "high_keywords":        high_hits,
        "urgency_words":        urgency_hits,
        "sentence_count":       len(sentences),
        "avg_sentence_length":  round(avg_sentence_length, 1),
        "verb_ratio":           round(verb_ratio, 2),
        "negated_words":        negations,
        # Quick pre-classification hint for Claude
        "keyword_severity_hint": (
            "critical" if critical_hits else
            "high"     if high_hits     else
            "medium"   if urgency_hits  else
            "unknown"
        )
    }