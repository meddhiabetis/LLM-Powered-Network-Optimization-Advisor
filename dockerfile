# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Use pip cache for faster builds
RUN --mount=type=cache,target=/root/.cache/pip pip install --default-timeout=1000 -r requirements.txt

COPY . .

# Pre-download HuggingFace model to cache
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('unsloth/llama-3-8b-bnb-4bit'); AutoTokenizer.from_pretrained('unsloth/llama-3-8b-bnb-4bit')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]