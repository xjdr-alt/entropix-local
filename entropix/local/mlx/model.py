import math
import mlx.core as mx
import mlx.nn as nn

#global imports
from entropix.local.mlx.mlx_weights import LayerWeights, XfmrWeights
from entropix.local.config import ModelParams

#framework specific configs
from entropix.local.mlx.kvcache import KVCache
from entropix.local.mlx.metrics import AttnStats


DEFAULT_MASK_VALUE = -1e9

from typing import Tuple, Optional

def rms_norm(x: mx.array, w: mx.array, eps: float = 1e-5) -> mx.array:
  return w * (x * mx.rsqrt(mx.power(x, 2).mean(-1, keepdims=True) + eps))

def apply_rotary_emb(xq: mx.array, xk: mx.array, freqs_cis: mx.array, dtype=mx.float32) -> Tuple[mx.array, mx.array]:
    reshape_xq = mx.reshape(xq.astype(mx.float32), (*xq.shape[:-1], -1, 2))
    reshape_xk = mx.reshape(xk.astype(mx.float32), (*xk.shape[:-1], -1, 2))

    xq_ = mx.array(reshape_xq[..., 0] + 1j * reshape_xq[..., 1], dtype=mx.complex64)
    xk_ = mx.array(reshape_xk[..., 0] + 1j * reshape_xk[..., 1], dtype=mx.complex64)
    
    xq_out = xq_ * mx.expand_dims(mx.expand_dims(freqs_cis, 0), 2)
    xk_out = xk_ * mx.expand_dims(mx.expand_dims(freqs_cis, 0), 2)
    
    xq_out = mx.reshape(mx.stack([mx.real(xq_out), mx.imag(xq_out)], axis=-1), (*xq_out.shape[:-1], -1))
    xk_out = mx.reshape(mx.stack([mx.real(xk_out), mx.imag(xk_out)], axis=-1), (*xk_out.shape[:-1], -1))
    
    return xq_out.astype(dtype), xk_out.astype(dtype)

def attention(x: mx.array, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: mx.array, kvcache: KVCache, attn_mask: Optional[mx.array] = None) -> Tuple[mx.array, KVCache, mx.array]:
    bsz, _, _ = x.shape

    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = mx.reshape(mx.matmul(x, layer_weights.wq.T), (bsz, -1, model_params.n_local_heads, model_params.head_dim))
    xk = mx.reshape(mx.matmul(x, layer_weights.wk.T), (bsz, -1, model_params.n_local_kv_heads, model_params.head_dim))
    xv = mx.reshape(mx.matmul(x, layer_weights.wv.T), (bsz, -1, model_params.n_local_kv_heads, model_params.head_dim))

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)

    xq = mx.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = mx.transpose(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = mx.transpose(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
    
    scores = mx.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.astype(mx.float32)
    
    if cur_pos == 0:
        scores = scores + attn_mask
    
    mask = mx.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = mx.where(mask >= DEFAULT_MASK_VALUE * 0.5, scores, DEFAULT_MASK_VALUE)
    scores = mx.softmax(padded_logits, axis=-1).astype(x.dtype)
    
    output = mx.matmul(scores.astype(values.dtype), values)
    output = mx.reshape(mx.transpose(output, (0, 2, 1, 3)), (xq.shape[0], xq.shape[2], -1))
    out = mx.matmul(output, layer_weights.wo.T)
    
    if len(x.shape) == 2:
        out = mx.squeeze(out, 0)
    
    return out, kvcache, pre_scores

def feed_forward(x: mx.array, layer_weights: LayerWeights) -> mx.array:
    h1 = mx.matmul(x, layer_weights.w1.T)
    s1 = nn.silu(h1)
    h3 = mx.matmul(x, layer_weights.w3.T)
    h = s1 * h3
    return mx.matmul(h, layer_weights.w2.T)

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: mx.array, cur_pos: int, freqs_cis: mx.array, kvcache: KVCache, attn_mask: Optional[mx.array]=None) -> Tuple[mx.array, KVCache, mx.array, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    
    logits = mx.matmul(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)
    return logits, kvcache, scores, attn_stats