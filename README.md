# Emergency Call Severity Analyzer

A REST API that transcribes recorded emergency calls and classifies their severity. The service combines speech-to-text (Whisper), a fine-tuned DistilBERT classifier, and rule-based NLP feature extraction (spaCy) to support faster prioritization of incoming calls.

> **Disclaimer:** This project is intended for research and educational purposes. It is not a certified emergency-response system and must not replace trained dispatchers or official emergency services.

---

## Overview

The Emergency Call Severity Analyzer accepts an audio recording, produces a transcript, and returns a severity classification with supporting evidence.

**Key capabilities**

- **Speech-to-text:** Audio is transcribed locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`tiny` model, CPU, int8 quantization).
- **Severity classification:** A fine-tuned DistilBERT model ([`chirantana20/emergency-severity-distilbert`](https://huggingface.co/chirantana20/emergency-severity-distilbert)) assigns one of five severity levels.
- **Explainable signals:** spaCy extracts critical keywords, urgency words, and negations (for example, "not breathing") that accompany each prediction.
- **Conservative escalation:** Low-confidence "low" and "non_emergency" predictions are automatically escalated to "medium", favoring a false alarm over a missed critical call.
- **Container-ready:** A Dockerfile is included for deployment on platforms such as Hugging Face Spaces or Render.

### Severity Levels

| Level           | Description                                        |
| --------------- | -------------------------------------------------- |
| `critical`      | Immediate threat to life (e.g., not breathing)     |
| `high`          | Serious emergency (e.g., fire, severe bleeding)    |
| `medium`        | Requires attention (e.g., suspected broken bone)   |
| `low`           | Minor issue (e.g., small cut)                      |
| `non_emergency` | Not an emergency (e.g., noise complaint)           |

---

## Architecture

```
Audio file --> Whisper (transcribe.py) --> Transcript
                                              |
                        +---------------------+---------------------+
                        |                                           |
                        v                                           v
            DistilBERT classifier                       spaCy NLP features
              (model/predict.py)                         (nlp_features.py)
                        |                                           |
                        +---------------------+---------------------+
                                              |
                                              v
                                   Severity + NLP signals
                                        (analyze.py)
                                              |
                                              v
                                        JSON response
```

---

## Project Structure

```
emergency-call/
├── app.py             # Flask application: health check and /analyze endpoint
├── transcribe.py      # Audio-to-text using faster-whisper
├── analyze.py         # Combines classifier output with NLP signals
├── nlp_features.py    # spaCy pipeline: keywords, urgency words, negations
├── model/
│   └── predict.py     # DistilBERT inference and safety escalation rule
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container configuration (port 7860)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- Internet access on first run (the Whisper and DistilBERT models are downloaded from the Hugging Face Hub)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Chirantana20/emergency-call.git
cd emergency-call

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the server
python app.py
```

The API is served at `http://localhost:7860`. The port can be overridden with the `PORT` environment variable.

### Docker

```bash
docker build -t emergency-call .
docker run -p 7860:7860 emergency-call
```

The container runs the application with Gunicorn (one worker, 120-second timeout) on port 7860.

---

## API Reference

### `GET /`

Health check endpoint.

**Response**

```json
{ "status": "ok" }
```

### `POST /analyze`

Analyzes an audio recording. The request must be sent as `multipart/form-data` with the audio file in a field named `audio`. The maximum upload size is 16 MB.

**Request**

```bash
curl -X POST http://localhost:7860/analyze \
  -F "audio=@call.webm"
```

**Response**

```json
{
  "transcript": "He's not breathing, I can't find a pulse, please hurry",
  "severity": {
    "level": "critical",
    "confidence": 0.97,
    "all_scores": {
      "critical": 0.97,
      "high": 0.02,
      "medium": 0.01,
      "low": 0.0,
      "non_emergency": 0.0
    },
    "flagged": null,
    "nlp_signals": {
      "critical_keywords": ["not breathing"],
      "urgency_words": ["please", "hurry"],
      "negations": ["breathing"]
    }
  }
}
```

*The values shown are illustrative.*

**Response fields**

| Field                          | Description                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| `transcript`                   | Text transcribed from the audio                                                               |
| `severity.level`               | Predicted severity level                                                                      |
| `severity.confidence`          | Model confidence for the predicted level (0 to 1)                                             |
| `severity.all_scores`          | Probability assigned to each severity level                                                   |
| `severity.flagged`             | `null` by default; contains an explanation if the prediction was escalated or too short to classify |
| `severity.nlp_signals`         | Critical keywords, urgency words, and negations detected in the transcript                    |

**Error responses**

| Status | Response body                              | Cause                             |
| ------ | ------------------------------------------ | --------------------------------- |
| `400`  | `{"error": "No audio file provided"}`      | The `audio` field is missing      |
| `400`  | `{"error": "Empty audio file"}`            | The uploaded file has no content  |
| `400`  | `{"error": "No speech detected"}`          | The transcript was empty          |
| `500`  | `{"error": "Internal processing error"}`   | Unexpected server-side failure    |

---

## Testing

`model/predict.py` includes a sanity check that runs the classifier against sample transcripts, including negation and mixed-signal cases:

```bash
python -m model.predict
```

---

## Technology Stack

| Area               | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| Web framework      | Flask, Flask-CORS, Gunicorn                             |
| Speech recognition | faster-whisper (Whisper `tiny`)                         |
| Classification     | Hugging Face Transformers, PyTorch (fine-tuned DistilBERT) |
| NLP                | spaCy (`en_core_web_sm`)                                |
| Deployment         | Docker (compatible with Hugging Face Spaces and Render) |

---

## Configuration

| Setting                | Location             | Notes                                                                 |
| ---------------------- | -------------------- | --------------------------------------------------------------------- |
| Whisper model size     | `transcribe.py`      | Replace `"tiny"` with `"base"` or larger to improve accuracy at the cost of speed |
| Transcription language | `transcribe.py`      | Currently fixed to English (`language="en"`)                          |
| Escalation threshold   | `model/predict.py`   | Confidence cut-off of `0.80` for low-severity predictions             |
| Maximum upload size    | `app.py`             | Controlled by `MAX_CONTENT_LENGTH` (default 16 MB)                    |
| Server port            | Environment variable | `PORT` (default `7860`)                                               |

---

## Limitations

- Supports English only.
- The `tiny` Whisper model may perform poorly on noisy audio or heavily accented speech.
- Keyword lists are manually curated and not exhaustive.
- Classification quality depends on the data used to fine-tune the model.
- The system has not been formally evaluated or approved for real-world emergency use.

---

## Future Enhancements

- Multilingual support
- Real-time and streaming audio analysis
- Location and entity extraction to assist dispatch
- Larger, more diverse training data and formal evaluation
- Web front-end for recording calls and viewing results
