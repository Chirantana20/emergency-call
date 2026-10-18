"""Container health probe. Exits 0 if the API answers on `/`, otherwise 1."""
import os
import sys
import urllib.request

port = os.environ.get("PORT", "7860")

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
