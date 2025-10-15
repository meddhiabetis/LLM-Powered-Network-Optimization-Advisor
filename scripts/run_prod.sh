#!/usr/bin/env bash
# Run the API in prod profile (8B 4-bit + LoRA adapter).
# Ensure a capable GPU, bitsandbytes installed, and adapter files present.
set -euo pipefail

export MODEL_PROFILE=${MODEL_PROFILE:-prod}
export ADAPTER_PATH=${ADAPTER_PATH:-"app/network_optimizer"}
export MODEL_CACHE=${MODEL_CACHE:-"./hf_models/cache"}

uvicorn app.main:app --host 0.0.0.0 --port 8000