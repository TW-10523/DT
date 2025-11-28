# app/models/model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

MODEL_ID = os.environ.get("MODEL_ID", "YourOrg/gpt-oss-20b")
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "./offload")
LOCAL_DIR = os.environ.get("LOCAL_MODEL_DIR", "./local_models/gpt_oss_20b")

_tokenizer = None
_model = None

def load_model(local=True, device_map="auto", dtype=torch.bfloat16):
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    model_src = LOCAL_DIR if local and os.path.isdir(LOCAL_DIR) else MODEL_ID

    _tokenizer = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_src,
        torch_dtype=dtype,
        device_map=device_map,
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    _model.eval()
    return _model, _tokenizer
