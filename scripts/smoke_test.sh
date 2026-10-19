#!/usr/bin/env bash
# Smoke-test a running instance of the API.
#
# Usage: scripts/smoke_test.sh [BASE_URL] [AUDIO_FILE]
#   BASE_URL    defaults to http://localhost:7860
#   AUDIO_FILE  optional; enables the full end-to-end /analyze check
# Env:   WAIT_SECONDS  how long to wait for the API to come up (default 30)
#
# Exits non-zero on the first failed check, so it can be used in CI/CD pipelines.
set -euo pipefail

BASE_URL="${1:-http://localhost:7860}"
BASE_URL="${BASE_URL%/}"
AUDIO_FILE="${2:-}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1" >&2; exit 1; }

# Print only the HTTP status code (000 means the connection itself failed).
http_status() { curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@" || true; }

# --- 1. Wait for the service to answer, then check the health endpoint --------
echo "Testing ${BASE_URL}"
deadline=$((SECONDS + WAIT_SECONDS))
until [[ "$(http_status "${BASE_URL}/")" == "200" ]]; do
    if (( SECONDS >= deadline )); then
        fail "GET / did not return 200 within ${WAIT_SECONDS}s (is the server running?)"
    fi
    sleep 2
done
pass "GET / returns 200"

# --- 2. Input validation: a request with no file must be rejected -------------
status="$(http_status -X POST "${BASE_URL}/analyze")"
[[ "${status}" == "400" ]] || fail "POST /analyze without audio returned ${status} (expected 400)"
pass "POST /analyze without audio returns 400"

# --- 3. Optional end-to-end check with a real recording -----------------------
if [[ -z "${AUDIO_FILE}" ]]; then
    echo "SKIP  end-to-end /analyze check (pass an audio file as the 2nd argument to enable)"
    exit 0
fi

[[ -f "${AUDIO_FILE}" ]] || fail "audio file not found: ${AUDIO_FILE}"

response="$(curl -s --max-time 120 -w $'\n%{http_code}' -F "audio=@${AUDIO_FILE}" "${BASE_URL}/analyze")"
status="${response##*$'\n'}"
body="${response%$'\n'*}"

[[ "${status}" == "200" ]] || fail "POST /analyze returned ${status}: ${body}"

level="$(python3 -c 'import json, sys; print(json.load(sys.stdin)["severity"]["level"])' <<< "${body}")" \
    || fail "response is not valid JSON with severity.level: ${body}"

case "${level}" in
    critical | high | medium | low | non_emergency) pass "POST /analyze returned severity '${level}'" ;;
    *) fail "unexpected severity level: '${level}'" ;;
esac
