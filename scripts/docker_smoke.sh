#!/usr/bin/env bash
# Build the Docker image, start a container, smoke-test it, then clean up.
#
# Usage: scripts/docker_smoke.sh [AUDIO_FILE]
# Env:   IMAGE (default emergency-call), HOST_PORT (default 7860)
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-emergency-call}"
HOST_PORT="${HOST_PORT:-7860}"

echo "==> Building ${IMAGE}"
docker build -t "${IMAGE}" .

echo "==> Starting container on port ${HOST_PORT}"
container_id="$(docker run -d --rm -p "${HOST_PORT}:7860" "${IMAGE}")"

# Always stop the container, even if the smoke test fails
cleanup() {
    echo "==> Stopping container"
    docker stop "${container_id}" > /dev/null
}
trap cleanup EXIT

# Generous wait: the first request may still be loading models into memory
WAIT_SECONDS=120 scripts/smoke_test.sh "http://localhost:${HOST_PORT}" "${1:-}"
