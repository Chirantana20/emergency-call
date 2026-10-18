# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 (builder): install dependencies and cache model weights
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv /opt/venv
WORKDIR /build

# CPU-only PyTorch. The default PyPI wheel bundles ~2 GB of CUDA libraries
# that are useless on a CPU-only host. Installed first so it is cached as its own layer.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies change less often than code, so copy requirements.txt on its own
# to keep this layer cached across code-only rebuilds.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Download model weights at build time so containers start fast and never depend
# on the Hugging Face Hub being reachable at request time.
ARG WHISPER_MODEL_SIZE=tiny
ENV WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE}
COPY model/ model/
COPY transcribe.py .
COPY scripts/download_models.py scripts/download_models.py
RUN PYTHONPATH=/build python scripts/download_models.py

# ---------------------------------------------------------------------------
# Stage 2 (runtime): small image with only what is needed to serve requests
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    PATH="/opt/venv/bin:$PATH" \
    PORT=7860

# Must match the value used in the builder stage so the cached model is the one loaded.
ARG WHISPER_MODEL_SIZE=tiny
ENV WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE}

# Run as a non-root user. UID 1000 is also what Hugging Face Spaces expects.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/hf /opt/hf
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 7860

# python:slim has no curl, so the probe is a tiny Python script.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

# Shell form so ${PORT} is expanded (platforms like Cloud Run inject their own PORT);
# `exec` makes gunicorn PID 1 so it receives SIGTERM and shuts down cleanly.
# One worker: each worker would load its own copy of Whisper + DistilBERT into memory.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --timeout 120 --access-logfile - app:app"]
