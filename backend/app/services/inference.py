# app/models/inference.py
from ..models.model_loader import load_model
import torch

_model, _tokenizer = None, None

def init():
    global _model, _tokenizer
    _model, _tokenizer = load_model()
    return _model, _tokenizer

def generate(prompt, max_new_tokens=256, temperature=0.0):
    if _model is None:
        init()
    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k,v in inputs.items()}  # CPU/offload mode
    out = _model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    return _tokenizer.decode(out[0], skip_special_tokens=True)
