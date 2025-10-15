# Run the API in dev profile (small model, no adapter).
if (-not $env:MODEL_PROFILE) { $env:MODEL_PROFILE = "dev" }
$env:ADAPTER_PATH  = ""
if (-not $env:MODEL_CACHE) { $env:MODEL_CACHE = "./hf_models/cache" }

uvicorn app.main:app --host 0.0.0.0 --port 8000