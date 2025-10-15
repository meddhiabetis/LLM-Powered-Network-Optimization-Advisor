#!/usr/bin/env bash
# Run the API in dev profile (small model, no adapter).
set -euo pipefail

export MODEL_PROFILE=${MODEL_PROFILE:-dev}
export ADAPTER_PATH=${ADAPTER_PATH:-""}
export MODEL_CACHE=${MODEL_CACHE:-"./hf_models/cache"}

uvicorn app.main:app --host 0.0.0.0 --port 8000