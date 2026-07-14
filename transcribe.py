# transcribe.py — uses faster-whisper, no numba, no DLL issues
import tempfile
import os
from faster_whisper import WhisperModel

# Load once at startup — "tiny" is fastest, swap for "base" if accuracy is poor
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Whisper model...")
        # cpu + int8 = fastest on machines without GPU
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
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