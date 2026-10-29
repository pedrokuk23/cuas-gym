"""Shared helpers for the MLX (Apple silicon) scripts."""
import json
import os

from cuas_gym.prompt import SYSTEM, build_prompt


def chat_ids(tok, inst, attrs=True):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(inst, attrs=attrs)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return list(ids) if not isinstance(ids, str) else tok.encode(ids)


def load_model(model, adapter=None, lora_layers=None, lora_rank=None, lora_scale=None, lora_keys=None):
    """Loads the model; with an adapter folder, applies and loads it; with lora_layers, adds new LoRA layers."""
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
    m, tok = load(model)
    # Training scripts set CUAS_DTYPE=float32: with 16-bit activations (float16 or bfloat16) the backward pass
    # through the quantized layers gives gradient norms of 1e8-1e10 on the same examples where float32 gives ~10,
    # so LoRA training diverged. Generation keeps the native dtype (CUAS_DTYPE unset or "native").
    dt = os.environ.get("CUAS_DTYPE", "native")
    if dt != "native":
        import mlx.core as mx
        m.set_dtype(getattr(mx, dt))
    m.freeze()
    cfg = None
    if adapter:
        load_adapters(m, adapter)
        with open(os.path.join(adapter, "adapter_config.json")) as f:
            cfg = json.load(f)
    elif lora_layers:
        cfg = {"fine_tune_type": "lora", "num_layers": lora_layers,
               "lora_parameters": {"rank": lora_rank, "scale": lora_scale, "dropout": 0.0}}
        if lora_keys:
            cfg["lora_parameters"]["keys"] = list(lora_keys)
        linear_to_lora_layers(m, lora_layers, cfg["lora_parameters"])
    return m, tok, cfg


def save_adapter(model, cfg, folder):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    os.makedirs(folder, exist_ok=True)
    mx.save_safetensors(os.path.join(folder, "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    with open(os.path.join(folder, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
