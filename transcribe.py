# transcribe.py — uses faster-whisper, no numba, no DLL issues
import os
import tempfile

from faster_whisper import WhisperModel

# "tiny" is fastest; set WHISPER_MODEL_SIZE=base (or larger) if accuracy is poor.
# Also read by scripts/download_models.py so the Docker build caches the same model.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")

# Loaded lazily on first request, then reused
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Whisper model...")
        # cpu + int8 = fastest on machines without GPU
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("Whisper ready.")
    return _model


def transcribe_audio(audio_bytes: bytes) -> dict:
    model = get_model()

    # faster-whisper needs a file path
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, language="en")

        # segments is a generator — consume it fully
        segment_list = []
        full_text    = ""
        for seg in segments:
            full_text += seg.text + " "
            segment_list.append({
                "text":  seg.text,
                "start": round(seg.start, 2),
                "end":   round(seg.end,   2),
            })

        return {
            "transcript": full_text.strip(),
            "language":   info.language,
            "segments":   segment_list
        }

    finally:
        os.unlink(tmp_path)