import os
import torch
from transformers import AutoModel, AutoTokenizer

def try_accelerate_load(model_path, torch_dtype=torch.float16):
    """Attempt to load a sharded/fp16 model using accelerate.
    Returns (model, tokenizer) on success, raises on failure.

    This function intentionally keeps the accelerate-specific logic
    isolated so callers can fall back to non-accelerate loading.
    """
    try:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    except Exception as e:
        raise RuntimeError("accelerate not available") from e

    # Use init_empty_weights to avoid allocating full weights on CPU
    with init_empty_weights():
        # create an empty model skeleton (meta tensors)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    # Dispatch and load checkpoint into multiple devices automatically
    model = load_checkpoint_and_dispatch(
        model,
        model_path,
        device_map="auto",
        dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def model_device_set(model):
    """Return set of device strings used by model parameters.
    E.g. {'cuda:0', 'cuda:1'} or {'cpu'}.
    """
    try:
        devs = {str(p.device) for p in model.parameters()}
        return devs
    except Exception:
        return set()
