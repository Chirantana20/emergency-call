# Emergency Call Severity Analyzer

[![CI](https://github.com/Chirantana20/emergency-call/actions/workflows/ci.yml/badge.svg)](https://github.com/Chirantana20/emergency-call/actions/workflows/ci.yml)

A REST API that transcribes recorded emergency calls and classifies their severity. The service combines speech-to-text (Whisper), a fine-tuned DistilBERT classifier, and rule-based NLP feature extraction (spaCy) to support faster prioritization of incoming calls.

> **Disclaimer:** This project is intended for research and educational purposes. It is not a certified emergency-response system and must not replace trained dispatchers or official emergency services.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Development](#development)
- [Technology Stack](#technology-stack)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Product Roadmap](#product-roadmap)
- [Infrastructure Roadmap](#infrastructure-roadmap)
- [Author](#author)
- [License](#license)

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
├── app.py                  # Flask application: health check and /analyze endpoint
├── transcribe.py           # Audio-to-text using faster-whisper
├── analyze.py              # Combines classifier output with NLP signals
├── nlp_features.py         # spaCy pipeline: keywords, urgency words, negations
├── model/
│   └── predict.py          # DistilBERT inference and safety escalation rule
├── tests/                  # pytest suite (API, classifier, NLP features)
├── scripts/
│   ├── setup.sh            # Create a local dev environment
│   ├── smoke_test.sh       # Black-box check of a running API
│   ├── docker_smoke.sh     # Build image, run container, smoke-test, clean up
│   ├── download_models.py  # Pre-download model weights (used by the Dockerfile)
│   └── healthcheck.py      # Container health probe
├── .github/
│   ├── workflows/
│   │   └── ci.yml           # Lint, test, build and publish on every push/PR
│   ├── dependabot.yml       # Weekly updates for pinned GitHub Actions
│   └── pull_request_template.md
├── pyproject.toml          # pytest, coverage and Ruff configuration
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Runtime + test/lint dependencies
├── Dockerfile              # Multi-stage, non-root container image (port 7860)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- Internet access on first run (the Whisper and DistilBERT models are downloaded from the Hugging Face Hub)

### Local Installation

```bash
git clone https://github.com/Chirantana20/emergency-call.git
cd emergency-call

# Creates ./venv, installs dependencies and the spaCy model
scripts/setup.sh
source venv/bin/activate

# Start the server
python app.py
```

Manual alternative: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`.

The API is served at `http://localhost:7860`. The port can be overridden with the `PORT` environment variable.

### Docker

```bash
docker build -t emergency-call .
docker run -p 7860:7860 emergency-call
```

The image is built in two stages and is designed for production-style use:

- **CPU-only PyTorch**, which avoids roughly 2 GB of unused CUDA libraries.
- **Models baked in at build time** (Whisper and DistilBERT), so containers start quickly and do not depend on the Hugging Face Hub at request time.
- **Non-root user** (UID 1000) and a `HEALTHCHECK` that probes `GET /`.
- **Configurable port** through the `PORT` environment variable (default `7860`), which platforms such as Cloud Run set automatically.

The application runs under Gunicorn with one worker (each worker loads its own copy of the models) and a 120-second timeout.

To build, run, smoke-test, and clean up in one command:

```bash
scripts/docker_smoke.sh                 # optionally pass a path to an audio file
```

Every merge to `main` that passes CI is also published to GitHub Container Registry, so pulling the image requires no local build:

```bash
docker pull ghcr.io/chirantana20/emergency-call:latest
docker run -p 7860:7860 ghcr.io/chirantana20/emergency-call:latest
```

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

## Development

### Running the tests

```bash
pytest                                  # run the full suite
pytest --cov --cov-report=term-missing  # with coverage
ruff check .                            # lint
```

The suite is split by what each layer needs:

| File                        | Scope                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_api.py`         | Flask layer: validation, status codes, error handling (heavy modules are stubbed)  |
| `tests/test_predict.py`     | Softmax, label mapping, and the safety-escalation rule (fake model, no downloads)  |
| `tests/test_nlp_features.py`| spaCy keyword, urgency, and negation extraction                                    |
| `tests/test_analyze.py`     | Merging of classifier output and NLP signals                                       |

`model/predict.py` also contains a manual sanity check against the real model: `python -m model.predict`.

### Smoke-testing a running instance

```bash
scripts/smoke_test.sh                                    # health check + input validation
scripts/smoke_test.sh http://localhost:7860 call.webm    # plus an end-to-end analysis
```

The script exits non-zero on failure, so it is reused by the `docker` job in CI (below).

### Git workflow

- `main` is always deployable; all changes go through pull requests.
- Branch names use a type prefix: `feature/`, `fix/`, `chore/`, `docs/` (for example `fix/whole-word-keyword-matching`).
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
- Before opening a PR, make sure `pytest` and `ruff check .` pass, then complete the PR template.
- Enable branch protection on `main` (Settings → Branches) requiring the CI workflow to pass and the branch to be up to date before merging.

### Continuous Integration

Every push to `main` and every pull request runs the `CI` workflow (`.github/workflows/ci.yml`), in three stages that each depend on the previous one passing:

| Job      | What it does                                                                                     |
| -------- | -------------------------------------------------------------------------------------------------- |
| `lint`   | `ruff check .` — fails in seconds, before anything heavier runs                                    |
| `test`   | Installs CPU-only PyTorch and `requirements-dev.txt`, then runs the full `pytest` suite with coverage; the coverage report is uploaded as a build artifact |
| `docker` | Builds the real Docker image, runs it, and runs `scripts/smoke_test.sh` against it — the same check `scripts/docker_smoke.sh` runs locally |

On `main` only (never on pull requests), the `docker` job also pushes the built image to GitHub Container Registry, tagged `latest` and with the commit SHA, which is what `docker pull ghcr.io/chirantana20/emergency-call:latest` (above) actually pulls.

A few choices worth knowing about if you extend this workflow:

- **Actions are pinned to a full commit SHA**, not a version tag (e.g. `@11d5960a...` with `# v4` as a comment), because a tag can be moved to point at different code later but a commit SHA cannot — this is a standard supply-chain security practice for CI. `.github/dependabot.yml` opens a weekly PR to bump these pins, so they don't silently go stale.
- **Docker layers are cached** between runs with GitHub's own Actions cache (`cache-from`/`cache-to: type=gha`), so a code-only change reuses the slow dependency layers (PyTorch, the ML models) instead of rebuilding them every time.
- **`concurrency` cancels superseded runs** — pushing a second commit while the first is still building cancels the first, rather than wasting minutes on a result nobody will look at.

---

## Technology Stack

| Area               | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| Web framework      | Flask, Flask-CORS, Gunicorn                             |
| Speech recognition | faster-whisper (Whisper `tiny`)                         |
| Classification     | Hugging Face Transformers, PyTorch (fine-tuned DistilBERT) |
| NLP                | spaCy (`en_core_web_sm`)                                |
| Testing / quality  | pytest, pytest-cov, Ruff, ShellCheck                    |
| CI/CD              | GitHub Actions (lint, test, build, publish to GHCR)     |
| Deployment         | Docker (multi-stage, non-root; compatible with Hugging Face Spaces and Render) |

---

## Configuration

| Setting                | Location             | Notes                                                                 |
| ---------------------- | -------------------- | --------------------------------------------------------------------- |
| Whisper model size     | Environment variable | `WHISPER_MODEL_SIZE` (default `tiny`); use `base` or larger for accuracy at the cost of speed. For Docker, use `docker build --build-arg WHISPER_MODEL_SIZE=base .` so the matching model is cached in the image |
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

## Product Roadmap

- Multilingual support
- Real-time and streaming audio analysis
- Location and entity extraction to assist dispatch
- Larger, more diverse training data and formal evaluation
- Web front-end for recording calls and viewing results

---

## Infrastructure Roadmap

This project also doubles as a hands-on vehicle for production engineering practice. Each stage below builds on the last and is verified before moving to the next:

- [x] **Foundation** — automated tests (pytest), a production-style multi-stage Dockerfile, and developer scripts (`scripts/setup.sh`, `scripts/smoke_test.sh`, `scripts/docker_smoke.sh`)
- [x] **CI/CD** — a GitHub Actions pipeline (lint → test → Docker build and smoke test → publish to GHCR), with pull requests gated on all checks passing
- [ ] **Cloud deployment** — deploy the published GHCR image to a managed platform (e.g. Google Cloud Run)