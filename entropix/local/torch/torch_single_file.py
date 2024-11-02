from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

import torch
import torch.nn as nn
import torch.nn.functional as F

import ml_dtypes
import numpy as np

import math
import tyro

from pathlib import Path

from entropix.local.tokenizer import Tokenizer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')

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

class LayerWeights(NamedTuple):
  wq: torch.Tensor
  wk: torch.Tensor
  wv: torch.Tensor
  wo: torch.Tensor
  w1: torch.Tensor
  w2: torch.Tensor
  w3: torch.Tensor
  ffn_norm: torch.Tensor
  attention_norm: torch.Tensor

class XfmrWeights(NamedTuple):
  tok_embeddings: torch.Tensor
  norm: torch.Tensor
  output: torch.Tensor
  layer_weights: List[LayerWeights]


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
    min_p: float = 0.03

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

class AttnStats(NamedTuple):
    entropy: torch.Tensor  # (bsz, n_layers, num_heads)
    varentropy: torch.Tensor  # (bsz, n_layers, num_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            varentropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            n_layers=n_layers,
            n_heads=n_heads
        )

    @property
    def avg_entropy(self):
        return self.entropy.sum(dim=-1, keepdim=False)  # Average across heads

    @property
    def std_error(self):
        return torch.sqrt(torch.mean(self.varentropy)) / (self.n_heads * self.n_layers)

    def update(self, scores: torch.Tensor, layer_idx: int):
        probs = torch.nn.functional.softmax(scores, dim=-1)
        new_entropy = -torch.sum(torch.where(probs > 0, probs * torch.log(probs), torch.tensor(0.0)), dim=-1)
        new_varentropy = torch.sum(probs * (torch.log(probs) + new_entropy.unsqueeze(-1))**2, dim=-1)
        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy

        return self

class KVCache(nn.Module):
    def __init__(self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int):
        super(KVCache, self).__init__()
        self.register_buffer('k', torch.zeros((layers, bsz, max_seq_len, kv_heads, head_dim),dtype=torch.bfloat16,device=device))
        self.register_buffer('v', torch.zeros((layers, bsz, max_seq_len, kv_heads, head_dim),dtype=torch.bfloat16,device=device))

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        """Creates a new KVCache instance with initialized k and v tensors."""
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim)

    def update(self, xk: torch.Tensor, xv: torch.Tensor, layer_idx: int, cur_pos: int,n_rep: int):
        # Ensure xk and xv have the correct device and dtype
        xk = xk.to(self.k.dtype)
        xv = xv.to(self.v.dtype)

        insert_len = xk.size(1)
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xv

        if cur_pos == 0:
            keys = xk.repeat_interleave(n_rep, dim=2)
            values = xv.repeat_interleave(n_rep, dim=2)
        else:
            keys = self.k[layer_idx].repeat_interleave(n_rep, dim=2)
            values = self.v[layer_idx].repeat_interleave(n_rep, dim=2)

        return keys, values, self

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = F.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = torch.permute(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = torch.permute(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = torch.permute(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys.to(xq.dtype))
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = F.softmax(padded_logits, dim=-1)
    output = torch.matmul(scores.to(torch.float32), values.to(torch.float32))
    output = output.transpose(1, 2).reshape(xq.shape[0], xq.shape[2], -1).to(x.dtype)
    out = F.linear(output, layer_weights.wo)
    return out, kvcache, pre_scores

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
 return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
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
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats

# Constants
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

def calculate_varentropy_logsoftmax(logits: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the entropy and varentropy of the probability distribution using logsoftmax.
    """
    log_probs = F.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=dim) / LN_2  # Convert to base-2
    # For varentropy, broadcasting entropy to match dimensions
    entropy_expanded = entropy.unsqueeze(dim) if dim < 0 else entropy.unsqueeze(dim - logits.dim())
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy_expanded) ** 2, dim=dim)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Samples one token from a multinomial distribution with sorted probabilities.
    Implements the Gumbel-Max trick.
    """
    # Sample Gumbel noise
    # torch.distributions doesn't have a direct exponential sampler, so use uniform and transform
    uniform_noise = torch.rand(probs_sort.shape, device=device)
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
    # Equivalent to q sampled from exponential: using inverse transform
    # But using Gumbel-Max trick here
    sample = torch.argmax(probs_sort.to(device) + gumbel_noise.to(device), dim=-1, keepdim=True)
    return sample.long()


def _sample(logits, temperature, top_p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculate various metrics from logits and attention scores.
    """
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, min=1e-10, max=1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=1)

    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": entropy.mean(),
        "logits_varentropy": varentropy.mean(),
        "attn_entropy": attn_entropy.mean(),
        "attn_varentropy": attn_varentropy.mean(),
        "agreement": agreement.mean(),
        "interaction_strength": interaction_strength.mean()
    }

def sample(
    gen_tokens: torch.Tensor,
    logits: torch.Tensor,
    attention_scores: torch.Tensor,
    cfg: SamplerConfig,
    clarifying_question_token: int = 2564,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, Tuple[int, int, int], str]:
    """
    Main sampling function that selects the next token based on metrics and configuration.
    Returns the sampled token and the associated color formatting.
    """
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]
    return _sample(
                logits,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            ).to(device)


def compare_outputs(torch_output: torch.Tensor, jax_output: jax.Array, atol: float = 1e-5, rtol: float = 1e-8) -> None:
  jax_output_np = np.array(jax_output)
  torch_output_np = torch_output.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)

  try:
    np.testing.assert_allclose(torch_output_np, jax_output_np, atol=atol, rtol=rtol)
  except AssertionError as e:
    print(f'JAX output (first 30): {jax_output_np.flatten()[:30]}')
    print(f'PyTorch output (first 30): {torch_output_np.flatten()[:30]}')
    raise e

def load_weights(ckpt_dir: Path = Path('weights/1B-Instruct'), n_layers: int = 16):
  w = {}
  layer_weights = []
  with torch.inference_mode():
    for file in ckpt_dir.glob("*.npy"):
      name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
      jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
      np_weight = np.array(jax_weight).astype(np.float32)
      weight = torch.from_numpy(np_weight).to(torch.bfloat16).to(device)
      compare_outputs(torch_output=weight, jax_output=jax_weight)
      w[name] = weight.to(device)

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

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)

    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32)
  return mask


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

def main():
  with torch.inference_mode():
    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights()

    tokenizer = Tokenizer('entropix/tokenizer.model')
    raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')

    def generate(xfmr_weights, model_params, tokens):
      gen_tokens = None
      cur_pos = 0
      tokens = torch.tensor([tokens], dtype=torch.long).to(device)
      bsz, seqlen = tokens.shape
      attn_mask = build_attn_mask(seqlen, cur_pos).to(device)
      freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope).to(device)
      kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(device)
      logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
      next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
      gen_tokens = next_token
      print(tokenizer.decode([next_token.item()]), end='', flush=True)
      cur_pos = seqlen
      stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)
      cfg = SamplerConfig()
      while cur_pos < 8192:
        cur_pos += 1
        logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
        next_token = sample(gen_tokens, logits, scores, cfg=cfg)
        gen_tokens = torch.cat((gen_tokens.to(device), next_token.to(device)), dim=1)
        out_token = tokenizer.decode(next_token.tolist()[0])
        print(out_token, end='', flush=True)
        if torch.isin(next_token.to(device), stop.to(device)).any():
          break

    with torch.inference_mode():
      print(prompt)
      generate(xfmr_weights, model_params, raw_tokens1)

if __name__ == '__main__':
  tyro.cli(main)