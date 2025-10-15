# LLM-Powered Network Optimization Advisor

A containerized FastAPI inference service that provides network optimization recommendations using Large Language Models (LLMs), with optional LoRA fine-tuning.

## Profiles

| Profile | Base Model (default) | Adapter | Quantization | Intended Hardware |
|--------:|-----------------------|---------|--------------|-------------------|
| dev     | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | None (or optional tiny adapter) | none | Local (e.g., GTX 1650) |
| prod    | unsloth/llama-3-8b-bnb-4bit        | LoRA in `app/network_optimizer` | 4-bit | Larger GPU (≥ 12 GB VRAM recommended) |

The same codebase supports both profiles via environment variables (see `.env.example`).

## Endpoints

- GET `/health` — service and model status (profile, device, adapter flag)
- POST `/predict` — JSON body `{ "instruction": "...", "input": "..." }` → `{ "prediction": "..." }`

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Optimize network throughput under latency constraints","input":"RTT=40ms loss=0.3% bw=250Mbps"}'
```

---

## Quick Start (Dev)

```bash
python -m venv .venv311
source .venv311/bin/activate  # Windows: .\.venv311\Scripts\Activate
pip install --upgrade pip
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/health

Optional: copy `.env.example` → `.env` to store your defaults (e.g., `MODEL_PROFILE=dev`).

---

## Production (Example)

Requirements:
- Capable GPU (≥ 12 GB VRAM recommended)
- `bitsandbytes` and `accelerate` installed
- LoRA adapter placed in `app/network_optimizer/` (not committed to git)
- Hugging Face token if required for the base model

Setup:

```bash
python -m venv prodenv
source prodenv/bin/activate
pip install --upgrade pip
pip install -r requirements-prod.txt

export MODEL_PROFILE=prod
export ADAPTER_PATH=app/network_optimizer
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Docker

Dev (default):
```bash
docker compose up --build
# or with explicit dev override
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Prod:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

Enable GPU for Docker Desktop (Compose v2) by uncommenting `gpus: all` on compatible systems.

---

## LoRA Adapter

Place your adapter weights (e.g., `adapter_model.safetensors`, `adapter_config.json`) in:
```
app/network_optimizer/
```
This directory is ignored by git to avoid committing large or proprietary artifacts.

---

## Notebook: Fine‑Tuning Workflow

The repository includes a Jupyter notebook for training/fine‑tuning the model with LoRA:

- File: `notebook/llm-powered-network-optimization-advisor.ipynb`
- Goal: Fine‑tune Llama‑3‑8B (4‑bit quantized) for network optimization/anomaly use cases using Unsloth + LoRA.
- Output: A LoRA adapter (e.g., `adapter_model.safetensors` + `adapter_config.json`) that you can place under `app/network_optimizer/` for the prod profile.

### What the notebook does

1. Environment setup
   - Installs CUDA‑enabled PyTorch and complementary packages (`unsloth`, `datasets`, `trl`, etc.).
   - Designed to run on a GPU environment (e.g., Kaggle, Colab, or a Linux GPU VM). Windows WSL2 with CUDA also works.

2. Configuration
   - Sets base model to `unsloth/llama-3-8b-bnb-4bit` and typical LoRA hyperparameters:
     - Rank `r=16`, target modules across attention and MLP blocks, `lora_alpha=16`, `lora_dropout=0`.
   - Uses Unsloth’s `FastLanguageModel` with `load_in_4bit=True` for efficient VRAM usage.

3. Data preprocessing
   - Loads training/testing CSVs (in the example, from Kaggle competition paths).
   - Builds instruction/input/output triplets from network KPI columns.
   - Formats data into an Alpaca‑style prompt:
     ```
     Below is an instruction ...
     ### Instruction:
     {instruction}

     ### Input:
     {input}

     ### Response:
     {output}
     ```
   - Maps datasets to a `text` field ready for supervised fine-tuning.

4. Training
   - Initializes the 8B 4‑bit model and applies LoRA.
   - Trains with small batch sizes/gradient accumulation to fit common T4/A10 class GPUs.
   - Uses either TRL’s `SFTTrainer` or Unsloth training utilities (depending on your cell configuration).
   - Produces a LoRA adapter that captures the fine‑tuning delta (small size, fast to load).

5. Exporting the adapter
   - After training, save/export the LoRA adapter files (e.g., with `model.save_pretrained(<adapter_dir>)` or Unsloth export utility).
   - Copy the resulting adapter directory contents to:
     ```
     app/network_optimizer/
     ```
     The API will pick it up automatically in `prod` profile.

### How to run the notebook

- GPU environment required:
  - Colab (Pro recommended), Kaggle (T4), or Linux VM with NVIDIA GPU (≥ 12 GB VRAM recommended).
- Python and CUDA:
  - Use a PyTorch build matching your CUDA runtime (e.g., cu121).
  - The notebook cells include commands to install appropriate Torch wheels on Colab/Kaggle.
- Dataset paths:
  - The sample notebook uses Kaggle input paths (e.g., `/kaggle/input/.../Train.csv`).
  - Replace with your data paths if running elsewhere.

### Reproducibility tips

- Pin critical package versions (`transformers`, `unsloth`, `torch`, `trl`) to known‑working versions for your environment.
- Log training arguments and seed for deterministic runs where possible.
- Save the exact adapter and keep a note of the base model commit hash.

### Using the trained adapter in the API

1. Copy the adapter files into the API tree:
   ```
   app/network_optimizer/
     ├─ adapter_config.json
     ├─ adapter_model.safetensors
     └─ (other adapter files if any)
   ```
2. Run the API in `prod` profile:
   ```bash
   export MODEL_PROFILE=prod
   export ADAPTER_PATH=app/network_optimizer
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. The `/health` endpoint will show `adapter_active: true` if loaded successfully.

---

## Notes and Limitations

- The prod model (8B 4‑bit) typically requires significantly more VRAM than 4 GB for stable inference.
- If the base model is gated/private on Hugging Face, set `HF_TOKEN`.
- If you encounter quantization errors in prod, ensure `bitsandbytes` is installed and a CUDA‑enabled PyTorch build is used.

---

## Project Structure

```
app/
  config.py          # central configuration
  model_loader.py    # tokenizer/model/adapters loading
  main.py            # FastAPI app (health + predict)
  network_optimizer/ # LoRA adapter (not committed)
notebook/
  llm-powered-network-optimization-advisor.ipynb
scripts/             # convenience run scripts for dev/prod
Dockerfile
docker-compose*.yml
requirements-*.txt
.env.example
readme.md
```

---

## License

Add a LICENSE file if you want to specify usage terms (e.g., MIT).