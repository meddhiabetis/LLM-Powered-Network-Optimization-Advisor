"""
FastAPI application entrypoint.

Endpoints:
- GET  /health   -> service and model status
- POST /predict  -> generate text using instruction + input
"""

import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .model_loader import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("network-optimizer")

# Load settings and model bundle at startup
settings = get_settings()
bundle = load_model()

app = FastAPI(
    title="LLM-Powered Network Optimization Advisor",
    description="Dual-profile (dev/prod) FastAPI service for network optimization suggestions using LLMs + optional LoRA.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    instruction: str
    input: str


class PredictResponse(BaseModel):
    prediction: str


PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{inp}

### Response:
"""


@app.get("/health")
def health():
    """
    Lightweight liveness endpoint with model metadata.
    """
    return {
        "status": "ok",
        "profile": settings.model_profile,
        "base_model": settings.base_model,
        "adapter_active": bool(settings.adapter_path),
        "quantization": settings.quantization,
        "device": bundle.device,
        "max_new_tokens": settings.max_new_tokens,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Generate a response from the model using an instruction + input pattern.
    """
    try:
        prompt = PROMPT_TEMPLATE.format(instruction=req.instruction, inp=req.input)
        tokens = bundle.tokenizer(prompt, return_tensors="pt", truncation=True)
        tokens = {k: v.to(bundle.device) for k, v in tokens.items()}

        with torch.no_grad():
            output = bundle.model.generate(
                **tokens,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
            )

        decoded = bundle.tokenizer.decode(output[0], skip_special_tokens=True)
        if "### Response:" in decoded:
            decoded = decoded.split("### Response:")[-1].strip()

        return PredictResponse(prediction=decoded)

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e