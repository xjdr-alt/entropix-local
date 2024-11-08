import os
from typing import List, NamedTuple, Optional, Dict

import torch
import ml_dtypes
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import ml_dtypes
import mlx.core as mx

from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

#global imports
from entropix.local.config import MODEL_PATHS, MODEL_IDS, MODEL_CONFIGS, ModelConfig


def translate_key(in_key: str):
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        if out_key.endswith('input_layernorm'):
            out_key = out_key.replace('input_layernorm', 'attention_norm')
        elif out_key.endswith('mlp.down_proj'):
            out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
        elif out_key.endswith('mlp.gate_proj'):
            out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
        elif out_key.endswith('mlp.up_proj'):
            out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
        elif out_key.endswith('post_attention_layernorm'):
            out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        elif out_key.endswith('self_attn.k_proj'):
            out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
        elif out_key.endswith('self_attn.o_proj'):
            out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
        elif out_key.endswith('self_attn.q_proj'):
            out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
        elif out_key.endswith('self_attn.v_proj'):
            out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
        elif out_key.endswith('down_proj'):
            out_key = out_key.replace('down_proj', 'w2')
        elif out_key.endswith('gate_proj'):
            out_key = out_key.replace('gate_proj', 'w1')
        elif out_key.endswith('up_proj'):
            out_key = out_key.replace('up_proj', 'w3')
        elif out_key == 'embed_tokens':
            out_key = 'tok_embeddings'
        elif out_key == 'norm':
            out_key = 'norm'
        else:
            print(f"Don't know how to handle {in_key=}")
    elif out_key == 'lm_head':
        out_key = 'output'
    else:
        print(f"Don't know how to handle {in_key=}")
    return f'{out_key}.weight'

def reverse_permute(tensor: torch.Tensor, config: ModelConfig, is_kv: bool = False) -> torch.Tensor:
    """
    Reverse permute operation with adaptive dimensions based on model size and weight type.

    Args:
        tensor: Input tensor to be permuted
        config: ModelConfig object containing model parameters
        is_kv: Boolean flag indicating if the tensor is for key/value weights

    Returns:
        torch.Tensor: Permuted tensor
    """
    # Determine the number of heads and dimensions based on whether it's a KV weight
    if is_kv:
        n_heads = config.n_kv_heads
        dim1 = (config.dim // config.n_heads) * config.n_kv_heads
    else:
        n_heads = config.n_heads
        dim1 = config.dim

    dim2 = config.dim

    # Perform the reverse permutation
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def download_weights_torch(model_id: str, out_dir: Optional[Path] = None):
    """Download and save weights in PyTorch format."""
    if model_id not in MODEL_PATHS:
        raise ValueError(f"Invalid model size: {model_id}. Choose from: {list(MODEL_PATHS.keys())}")
    out_dir = Path(MODEL_PATHS[model_id])
     
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    model_name = MODEL_IDS[model_id]
    config = MODEL_CONFIGS[model_id]
    token = os.environ['TOKEN']

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            offload_folder="/tmp/offload",
            token=token,
        )

        with torch.no_grad():
            state_dict = hf_model.state_dict()
            for hf_name, param in state_dict.items():
                #print(f' {hf_name}: {param.shape=}')
                name = translate_key(hf_name)

                # Apply reverse permute for attention weights
                if name.endswith('wq.weight'):
                    param = reverse_permute(param, config, is_kv=False)
                elif name.endswith('wk.weight'):
                    param = reverse_permute(param, config, is_kv=True)

                # Convert to bfloat16 and save
                bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
                bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
                #print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
                jnp.save(f'{out_dir}/{name}.npy', bf16_out)

    # Cleanup
    del hf_model
    del state_dict

class LayerWeights(NamedTuple):
    wq: mx.array
    wk: mx.array
    wv: mx.array
    wo: mx.array
    w1: mx.array
    w2: mx.array
    w3: mx.array
    ffn_norm: mx.array
    attention_norm: mx.array

class XfmrWeights(NamedTuple):
    tok_embeddings: mx.array
    norm: mx.array
    output: mx.array
    layer_weights: List[LayerWeights]

def load_weights_mx(model_id: str, ckpt_dir: Optional[Path] = None) -> XfmrWeights:
    """Load weights in MLX format."""
    if model_id not in MODEL_PATHS:
        raise ValueError(f"Invalid model size. Choose from: {list(MODEL_PATHS.keys())}")
    ckpt_dir = Path(MODEL_PATHS[model_id])
    config = MODEL_CONFIGS[model_id]
    n_layers = config.n_layers
    w = {}
    layer_weights = []
    
    for file in ckpt_dir.glob("*.npy"):
        name = '.'.join(str(file).split('/')[-1].split('.')[:-1])

        jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
        np_weight = np.array(jax_weight).astype(np.float32)
        weight = mx.array(np_weight, dtype=mx.bfloat16)
        w[name] = weight

    for i in range(n_layers):
        layer_weights.append(LayerWeights(
            wq=w[f'layers.{i}.attention.wq.weight'],
            wk=w[f'layers.{i}.attention.wk.weight'],
            wv=w[f'layers.{i}.attention.wv.weight'],
            wo=w[f'layers.{i}.attention.wo.weight'],
            w1=w[f'layers.{i}.feed_forward.w1.weight'],
            w2=w[f'layers.{i}.feed_forward.w2.weight'],
            w3=w[f'layers.{i}.feed_forward.w3.weight'],
            ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
            attention_norm=w[f'layers.{i}.attention_norm.weight'],
        ))

    return XfmrWeights(
        tok_embeddings=w['tok_embeddings.weight'],
        norm=w['norm.weight'],
        output=w['output.weight'],
        layer_weights=layer_weights
    )