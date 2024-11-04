import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Dict, NamedTuple
from enum import Enum

LN_2 = 0.69314718056 
DEFAULT_MASK_VALUE = -1e9

def calculate_attention_varentropy(attention_scores: mx.array, current_pos: mx.array) -> Tuple[mx.array, mx.array]:
    """Calculate the entropy and varentropy of attention probabilities with causal masking."""
    seq_length = attention_scores.shape[-1]
    mask = mx.arange(seq_length) >= current_pos
    mask = mask.reshape(1, 1, 1, -1)
    attention_scores = mx.where(mask, mx.array(DEFAULT_MASK_VALUE), attention_scores)
    
    attention_probs = mx.softmax(attention_scores, axis=-1)
    attention_probs_clamped = mx.clip(attention_probs, 1e-10, 1.0)
    entropy = -mx.sum(attention_probs * mx.log2(attention_probs_clamped), axis=-1)
    varentropy = mx.sum(attention_probs * (mx.log2(attention_probs_clamped) + mx.expand_dims(entropy, -1))**2, axis=-1)
    
    return entropy, varentropy, attention_probs

def calculate_varentropy_logsoftmax(logits: mx.array, axis: int = -1) -> Tuple[mx.array, mx.array]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = nn.log_softmax(logits, axis=axis)
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = mx.sum(probs * (log_probs / LN_2 + mx.expand_dims(entropy, -1))**2, axis=axis)
    return entropy, varentropy


def calculate_metrics(logits: mx.array, attention_scores: mx.array, current_pos: mx.array) -> Dict[str, mx.array]:
    logits_entropy, logits_varentropy = calculate_varentropy_logsoftmax(logits)
    attn_entropy, attn_varentropy, attention_probs = calculate_attention_varentropy(attention_scores, current_pos)
    mean_attention = mx.mean(attention_probs, axis=1)
    agreement = mx.mean(mx.abs(attention_probs - mx.expand_dims(mean_attention, 1)), axis=(1, 2))
    interaction_strength = mx.mean(mx.abs(attention_scores), axis=(1, 2, 3))

    return {
        "logits_entropy": mx.mean(logits_entropy),
        "logits_varentropy": mx.mean(logits_varentropy),
        "attn_entropy": mx.mean(attn_entropy),
        "attn_varentropy": mx.mean(attn_varentropy),
        "agreement": mx.mean(agreement),
        "interaction_strength": interaction_strength
    }

class AttnStats(NamedTuple):
    entropy: mx.array  # (bsz, n_layers, num_heads)
    varentropy: mx.array  # (bsz, n_layers, num_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=mx.zeros((bsz, n_layers, n_heads)),
            varentropy=mx.zeros((bsz, n_layers, n_heads)),
            n_layers=n_layers,
            n_heads=n_heads
        )

    @property
    def avg_entropy(self):
        return mx.sum(self.entropy, axis=-1)  # Average across heads

    @property
    def std_error(self):
        return mx.sqrt(mx.mean(self.varentropy)) / (self.n_heads * self.n_layers)

    def update(self, scores: mx.array, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = mx.softmax(scores, axis=-1)
        new_entropy = -mx.sum(mx.where(probs > 0, probs * mx.log(probs), mx.array(0.0)), axis=-1)
        new_varentropy = mx.sum(probs * (mx.log(probs) + mx.expand_dims(new_entropy, -1))**2, axis=-1)

        # Update entropy and varentropy tensors
        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy

        return self