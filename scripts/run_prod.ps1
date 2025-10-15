# Run the API in prod profile (8B 4-bit + LoRA adapter).
# Ensure a capable GPU, bitsandbytes installed, and adapter files present.
if (-not $env:MODEL_PROFILE) { $env:MODEL_PROFILE = "prod" }
if (-not $env:ADAPTER_PATH) { $env:ADAPTER_PATH = "app/network_optimizer" }
if (-not $env:MODEL_CACHE)  { $env:MODEL_CACHE  = "./hf_models/cache" }

uvicorn app.main:app --host 0.0.0.0 --port 8000