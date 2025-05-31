import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL_BASE = "unsloth/llama-3-8b-bnb-4bit"
ADAPTER_PATH = "network_optimizer"
MODEL_CACHE = "./hf_models/llama-3-8b-bnb-4bit"

def ensure_base_model():
    # Download base model if not present in cache
    if not os.path.exists(MODEL_CACHE):
        os.makedirs(MODEL_CACHE, exist_ok=True)
    # Try to load, will download if not in cache
    _ = AutoModelForCausalLM.from_pretrained(MODEL_BASE, cache_dir=MODEL_CACHE)
    _ = AutoTokenizer.from_pretrained(MODEL_BASE, cache_dir=MODEL_CACHE)

ensure_base_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE)
model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = FastAPI()

class Query(BaseModel):
    instruction: str
    input: str

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

@app.post("/predict")
def predict(query: Query):
    prompt = alpaca_prompt.format(query.instruction, query.input, "")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = pred_text.split("### Response:")[-1].strip()
    return {"prediction": response}