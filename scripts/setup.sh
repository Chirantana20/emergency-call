#!/usr/bin/env bash
# Set up a local development environment.
#
# Usage: scripts/setup.sh
# Env:   PYTHON=python3.12 scripts/setup.sh   (choose the interpreter)
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" > /dev/null 2>&1; then
    echo "error: '$PYTHON' not found. Install Python 3.11+ or set PYTHON=/path/to/python." >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "error: Python 3.11 or newer is required (found $("$PYTHON" --version))." >&2
    exit 1
fi

if [[ ! -d venv ]]; then
    echo "==> Creating virtual environment in ./venv"
    "$PYTHON" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

echo "==> Downloading spaCy language model"
python -m spacy download en_core_web_sm

echo
echo "Setup complete. Next steps:"
echo "  source venv/bin/activate"
echo "  pytest             # run the tests"
echo "  ruff check .       # lint"
echo "  python app.py      # start the API on http://localhost:7860"
