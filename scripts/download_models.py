"""Download and cache all model weights.

Run during `docker build` so the weights are baked into the image. It can also be run
locally (`python scripts/download_models.py`) to warm the Hugging Face cache before
the first request.
"""
from faster_whisper import WhisperModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model.predict import SAVE_PATH
from transcribe import WHISPER_MODEL_SIZE


def main() -> None:
    print(f"Downloading Whisper model '{WHISPER_MODEL_SIZE}'...")
    WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    print(f"Downloading classifier '{SAVE_PATH}'...")
    AutoTokenizer.from_pretrained(SAVE_PATH)
    AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)

    print("All models cached.")


if __name__ == "__main__":
    main()
