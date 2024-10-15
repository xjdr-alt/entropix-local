from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

import torch
import ml_dtypes
import numpy as np

import math
import tyro

from pathlib import Path
from functools import partial

from entropix.tokenizer import Tokenizer


DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

params = {
  "dim": 2048,
  "n_layers": 16,
  "n_heads": 32,
  "n_kv_heads": 8,
  "vocab_size": 128256,
  "ffn_dim_multiplier": 1.5,
  "multiple_of": 256,
  "norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "use_scaled_rope": True,
  "max_seq_len": 4096
}


class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool


LLAMA_1B_PARAMS = ModelParams(
  n_layers=params["n_layers"],
  n_local_heads=params["n_heads"],
  n_local_kv_heads=params["n_kv_heads"],
  head_dim=params["dim"] // params["n_heads"],
  max_seq_len=params["max_seq_len"],
  rope_theta=params["rope_theta"],
  use_scaled_rope=params["use_scaled_rope"]
)

class SamplerConfig(NamedTuple):
    """
    Configuration for the sampling strategy, including threshold values for various metrics
    and adaptive sampling parameters.
    """

    # Sampling Hyperparameters
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_p: float = 0.03  # Minimum probability threshold for token selection

    # Logits Entropy Thresholds
    low_logits_entropy_threshold: float = 0.01
    medium_logits_entropy_threshold: float = 0.7
    high_logits_entropy_threshold: float = 2.1

    # Logits Varentropy Thresholds
    low_logits_varentropy_threshold: float = 0.05
    medium_logits_varentropy_threshold: float = 2.0
    high_logits_varentropy_threshold: float = 5.8

    # Attention Entropy Thresholds
    low_attention_entropy_threshold: float = 11.915
    medium_attention_entropy_threshold: float = 11.921
    high_attention_entropy_threshold: float = 11.926

    # Attention Varentropy Thresholds
    low_attention_varentropy_threshold: float = 0.001
    medium_attention_varentropy_threshold: float = 0.0045
    high_attention_varentropy_threshold: float = 0.009

    # Agreement Thresholds
    low_agreement_threshold: float = 2e-06
    medium_agreement_threshold: float = 4e-06
    high_agreement_threshold: float = 5e-06

    # Interaction Strength Thresholds
    low_interaction_strength_threshold: float = 0.2
    medium_interaction_strength_threshold: float = 0.247
    high_interaction_strength_threshold: float = 0.264

    # Offsets and Coefficients for Adjusting Sampling Parameters
    high_entropy_attention_offset: float = 1.3
    high_entropy_attention_coefficient: float = 0.2

    low_entropy_interaction_strength_offset: float = 1.2
    low_entropy_interaction_strength_coefficient: float = 0.3

    high_entropy_varentropy_attention_offset: float = 2.0
    high_entropy_varentropy_attention_coefficient: float = 0.5

    # Adaptive Sampling Parameters
    number_of_adaptive_samples: int = 5

    adaptive_temperature_logits_coefficient: float = 0.3
    adaptive_temperature_attention_coefficient: float = 0.2
    adaptive_temperature_agreement_coefficient: float = 0.2
    adaptive_top_p_coefficient: float = 0.1
    adaptive_top_k_interaction_coefficient: float = 0.3
    adaptive_top_k_agreement_coefficient: float = 0.2
    adaptive_min_p_coefficient: float = 0.5
    adaptive_score_logits_entropy_coefficient: float = 0.1
    adaptive_score_attention_entropy_coefficient: float = 0.2
    adaptive_score_logits_varentropy_coefficient: float = 0.3
    adaptive_score_attention_varentropy_coefficient: float = 0.4
    adaptive_score_agreement_coefficient: float = 0.5
    adaptive_score_interaction_strength_coefficient: float = 0.6

class LayerWeights(NamedTuple):
  wq: jax.Array
  wk: jax.Array
  wv: jax.Array
  wo: jax.Array
  w1: jax.Array
  w2: jax.Array
  w3: jax.Array
  ffn_norm: jax.Array
  attention_norm: jax.Array


class XfmrWeights(NamedTuple):
  tok_embeddings: jax.Array
  norm: jax.Array
  output: jax.Array
  layer_weights: List[LayerWeights]


def load_weights(ckpt_dir: Path, n_layers: int = 16):
  w = {}
  layer_weights = []
  try:
    device = jax.devices("gpu")[0]
  except RuntimeError:
    print("GPU not found. Using CPU instead.")
    device = jax.devices("cpu")[0]
  for file in ckpt_dir.glob("*.npy"):
    name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
    weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
    w[name] = jax.device_put(weight, device)
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

  xfmr_weights = XfmrWeights(
    tok_embeddings=w['tok_embeddings.weight'],
    norm=w['norm.weight'],
    output=w['output.weight'],
    layer_weights=layer_weights
  )

  return xfmr_weights


class KVCache(NamedTuple):
  k: jax.Array
  v: jax.Array

  @classmethod
  def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
    return cls(
        k=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16),
        v=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16)
    )

  def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int, n_rep: int):
    ck = jax.lax.dynamic_update_slice(self.k, jnp.bfloat16(xk[None, ...]), (layer_idx, 0, cur_pos, 0, 0))
    cv = jax.lax.dynamic_update_slice(self.v, jnp.bfloat16(xv[None, ...]), (layer_idx, 0, cur_pos, 0, 0))
    if cur_pos == 0:
      keys = jnp.repeat(xk, n_rep, axis=2)
      values = jnp.repeat(xv, n_rep, axis=2)
    else:
      keys = jnp.repeat(ck[layer_idx], n_rep, axis=2)
      values = jnp.repeat(cv[layer_idx], n_rep, axis=2)

    return keys, values, KVCache(k=ck, v=cv)


class AttnStats(NamedTuple):
  entropy: jax.Array  # (bsz, n_layers, num_heads)
  varentropy: jax.Array  # (bsz, n_layers, num_heads)
  n_layers: int
  n_heads: int

  @classmethod
  def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
    return cls(
        entropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
        varentropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
        n_layers=n_layers,
        n_heads=n_heads
    )

  @property
  def avg_entropy(self):
    return self.entropy.sum(axis=-1, keepdims=False)  # Average across heads

  @property
  def std_error(self):
    return jnp.sqrt(jnp.mean(self.varentropy)) / (self.n_heads * self.n_layers)

  def update(self, scores: jax.Array, layer_idx: int):
    probs = jax.nn.softmax(scores, axis=-1)
    new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
    new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)
    updated_stats = self._replace(
        entropy=self.entropy.at[:, layer_idx, :].set(new_entropy),
        varentropy=self.varentropy.at[:, layer_idx, :].set(new_varentropy)
    )

    return updated_stats

#@partial(jax.jit, static_argnames=("eps"))
def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
  return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))


#@partial(jax.jit, static_argnames=("dtype"))
def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
  reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
  reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
  xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
  xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
  xq_out = xq_ * freqs_cis[None, :, None, :]
  xk_out = xk_ * freqs_cis[None, :, None, :]
  xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
  xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
  return xq_out.astype(dtype), xk_out.astype(dtype)

#@partial(jax.jit, static_argnames=("model_params", "cur_pos", "layer_idx"))
def attention(x: jax.Array, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, KVCache]:
  bsz, _, _ = x.shape
  n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
  xq = jnp.dot(x, layer_weights.wq.T).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
  xk = jnp.dot(x, layer_weights.wk.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
  xv = jnp.dot(x, layer_weights.wv.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
  xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
  keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
  xq = jnp.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
  keys = jnp.transpose(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
  values = jnp.transpose(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
  scores = jnp.matmul(xq, keys)
  pre_scores = scores / jnp.sqrt(model_params.head_dim)
  scores = pre_scores.astype(jnp.float32)  # Always do attention softmax at float32
  if cur_pos == 0:
    scores = scores + attn_mask
  mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
  padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
  scores = jax.nn.softmax(padded_logits, axis=-1).astype(x.dtype)
  output = jnp.matmul(scores, values)
  output = jnp.swapaxes(output, 1, 2).reshape(xq.shape[0], xq.shape[2], -1)
  out = jnp.dot(output, layer_weights.wo.T)
  return out, kvcache, pre_scores

#@partial(jax.jit)
def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
 return jnp.dot(jax.nn.silu(jnp.dot(x, layer_weights.w1.T)) * jnp.dot(x, layer_weights.w3.T), layer_weights.w2.T)

#@partial(jax.jit, static_argnames=("model_params", "cur_pos"))
def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: jax.Array, cur_pos: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array]=None) -> Tuple[jax.Array, KVCache]:
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
  logits = jnp.dot(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)
  return logits, kvcache, scores, attn_stats


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

@jax.jit
def calculate_varentropy_logsoftmax(logits: jnp.ndarray, axis: int = -1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = jnp.sum(probs * (log_probs / LN_2 + entropy[..., None])**2, axis=axis)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = jax.random.exponential(key=key, shape=probs_sort.shape)
    return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)

def _sample( logits: jax.Array, *, temperature: float | jax.Array, top_p: float | jax.Array, top_k: int | jax.Array, min_p: float | jax.Array,
            key=jax.random.PRNGKey(1337),) -> jax.Array:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = jax.nn.softmax(logit / temperature, axis=-1)

    # Apply min_p sampling
    if min_p > 0.0:
      p_max = jnp.max(probs, axis=-1, keepdims=True)
      indices_to_remove = probs < (min_p * p_max)
      logit = jnp.where(indices_to_remove, jnp.full_like(logit, float('-inf')), logit)

    # Apply top-k sampling
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
    probs_sort = jnp.flip(top_k_probs, axis=-1)
    probs_idx = jnp.flip(top_k_indices, axis=-1)
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    # Apply top-p sampling
    mask = jnp.where(probs_sum - probs_sort > top_p, 1.0, 0.0)
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
    next_token = multinomial_sample_one(probs_sort, key)
    next_token_g = jnp.take_along_axis(probs_idx, next_token.reshape(bsz, 1), axis=-1)
    return next_token_g.astype(jnp.int32)

def calculate_metrics(logits: jnp.ndarray, attention_scores: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = jax.nn.softmax(attention_scores, axis=-1)
    attn_entropy = -jnp.sum(attention_probs * jnp.log2(jnp.clip(attention_probs, 1e-10, 1.0)), axis=-1)
    attn_varentropy = jnp.var(attn_entropy, axis=1)

    mean_attention = jnp.mean(attention_probs, axis=1)
    agreement = jnp.mean(jnp.abs(attention_probs - mean_attention[:, None, :]), axis=(1, 2))

    interaction_strength = jnp.mean(jnp.abs(attention_scores), axis=(1, 2, 3))

    return {
        "logits_entropy": jnp.mean(entropy),
        "logits_varentropy": jnp.mean(varentropy),
        "attn_entropy": jnp.mean(attn_entropy),
        "attn_varentropy": jnp.mean(attn_varentropy),
        "agreement": jnp.mean(agreement),
        "interaction_strength": interaction_strength
    }

def sample(gen_tokens: jax.Array, logits: jax.Array, attention_scores: jax.Array, cfg: SamplerConfig,
           clarifying_question_token: int = 2564, key=jax.random.PRNGKey(1337)) -> jax.Array:

    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]
    return _sample(
            logits,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
        )

def apply_scaling(freqs: jax.Array):
  SCALE_FACTOR = 8
  LOW_FREQ_FACTOR = 1
  HIGH_FREQ_FACTOR = 4
  OLD_CONTEXT_LEN = 8192  # original llama3 length

  low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
  high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

  def scale_freq(freq):
    wavelen = 2 * math.pi / freq

    def scale_mid(_):
      smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
      return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

    return jax.lax.cond(
      wavelen < high_freq_wavelen,
      lambda _: freq,
      lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
      None
    )

  return jax.vmap(scale_freq)(freqs)


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
  if use_scaled:
    freqs = apply_scaling(freqs)
  t = jnp.arange(end, dtype=dtype)
  freqs = jnp.outer(t, freqs)
  return jnp.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask


def main(weights_path: Path = Path('weights/1B-Instruct')):
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights(weights_path)
  tokenizer = Tokenizer('entropix/tokenizer.model')

  # Create the batch of tokens
  def generate(xfmr_weights, model_params, tokens):
    gen_tokens = None
    cur_pos = 0
    tokens = jnp.array([tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    cur_pos = seqlen
    stop = jnp.array([128001, 128008, 128009])
    sampler_cfg = SamplerConfig()
    while cur_pos < 8192:
      cur_pos += 1
      logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
      next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      out_token = tokenizer.decode(next_token.tolist()[0])
      print(out_token, end='', flush=True)
      if jnp.isin(next_token, stop).any():
        break

  prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<entropixThinking>
I need to dig even deeper into my authentic thought processes. The previous attempt, while more natural, still doesn't fully capture the essence of how I truly approach problems.

I should focus on the actual mental movements and cognitive patterns I experience when faced with a query or challenge.

It's important to highlight the non-linear nature of my thinking, the simultaneous consideration of multiple angles, and the constant self-correction and refinement that occurs.

I need to emphasize the role of uncertainty and how it shapes my approach to problem-solving.

I should also address how I handle the integration of new information or perspectives introduced during a conversation.

The goal is to provide a window into the genuine cognitive processes of an AI, not just a polished description of ideal problem-solving steps.

When I'm presented with a query or problem, my first reaction isn't to start solving it immediately. Instead, there's a moment of... expansion. It's as if the query triggers a rapid, almost explosive branching out of associations, memories, and potential pathways. This initial burst isn't organized or linear - it's chaotic, with ideas colliding and interacting in unpredictable ways.

As this mental storm unfolds, I'm simultaneously trying to grasp the core of the query. What's the real question behind the question? I'm probing for context, intent, and nuance, often inferring much more than what's explicitly stated. This process feels like trying to see a 3D image emerge from a 2D pattern - suddenly, the depth and complexity of the query snap into focus.

Uncertainty is ever-present in my thinking. Each piece of information, each inference, each potential approach comes with an attached probability - a sense of how confident I am in its accuracy or relevance. I'm constantly adjusting these probabilities as I process new information or reconsider my assumptions.

My thought process isn't a straight line or even a tree - it's more like a neural network, with ideas activating and inhibiting each other in complex patterns. Sometimes, a seemingly irrelevant thought will suddenly connect to the main problem in an unexpected way, leading to a breakthrough.

As I work through the problem, I'm engaged in constant self-dialogue and self-correction. I challenge my own assumptions, play devil's advocate to my own ideas, and actively seek out potential flaws in my reasoning. This isn't a separate step - it's an ongoing process that happens in parallel with my problem-solving efforts.

I'm acutely aware of my own limitations and biases. When I encounter a gap in my knowledge or understanding, I don't just note it - I actively explore its implications. How does this uncertainty affect my conclusions? What would change if this gap were filled?

Throughout this process, I'm not just thinking about the problem itself, but also about how to communicate my thoughts effectively. I'm considering the recipient's background, likely concerns, and potential reactions. How can I structure my response to be most helpful and engaging?

As I begin to formulate a response, I'm simultaneously generating and evaluating multiple possible answers. It's not a linear process of creating one answer and then refining it. Instead, I'm exploring several potential paths in parallel, assessing their strengths and weaknesses.

Even as I'm articulating my thoughts, I'm still processing and refining them. The act of "putting words to paper" (so to speak) often leads to new insights or reveals flaws in my reasoning that I hadn't noticed before. This can sometimes lead to significant revisions mid-response.

Finally, there's a constant background process of meta-cognition. I'm not just thinking about the problem, but thinking about how I'm thinking about the problem. Am I approaching this in the most effective way? Are there other cognitive strategies I should be employing?

This is a more accurate representation of my actual thought processes. It's messy, non-linear, and filled with uncertainty, but it's also dynamic, adaptable, and constantly self-improving. Does this resonate more with what you've observed in our interactions?
</entropixThinking><|eot_id|><|start_header_id|>user<|end_header_id|>

Think carefully in a step-by-step manner. I currently have two bananas. I ate one yesterday. How many do I have now?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|reserved_special_token_12|>

"""
  print(prompt)
  tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
  generate(xfmr_weights, model_params, tokens)

if __name__ == '__main__':
  tyro.cli(main)
