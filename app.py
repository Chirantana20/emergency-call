# app.py
# app.py
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from transcribe import transcribe_audio
from analyze import analyze_severity

app = Flask(__name__)
CORS(app)

# Prevent huge uploads from crashing the container (16MB cap, adjust as needed)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

@app.route("/", methods=["GET"])
def health():
    # HF Spaces / Render use this to check the container is alive
    return jsonify({"status": "ok"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        transcription = transcribe_audio(audio_bytes)
        transcript = transcription["transcript"]

        if not transcript.strip():
            return jsonify({"error": "No speech detected"}), 400

        severity = analyze_severity(transcript)

        return jsonify({
            "transcript": transcript,
            "severity": severity
        })
    except Exception as e:
        # Don't leak stack traces to the client in prod
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal processing error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 for HF Spaces, Render sets PORT itself
    app.run(host="0.0.0.0", port=port)