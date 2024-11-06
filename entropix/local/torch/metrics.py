import torch
import torch.nn.functional as F
from typing import Tuple, Dict, NamedTuple
from enum import Enum

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

LN_2 = 0.69314718056 
DEFAULT_MASK_VALUE = -1e9

def calculate_attention_varentropy(attention_scores: torch.Tensor, current_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of attention probabilities with causal masking."""
    seq_length = attention_scores.shape[-1]
    device = attention_scores.device
    mask = torch.arange(seq_length, device=device) >= current_pos
    mask = mask.reshape(1, 1, 1, -1)
    attention_scores = torch.where(mask, torch.tensor(DEFAULT_MASK_VALUE, device=attention_scores.device), attention_scores)
    
    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs_clamped = torch.clamp(attention_probs, 1e-10, 1.0)
    entropy = -torch.sum(attention_probs * torch.log2(attention_probs_clamped), dim=-1)
    varentropy = torch.sum(attention_probs * (torch.log2(attention_probs_clamped) + entropy.unsqueeze(-1))**2, dim=-1)
    
    return entropy, varentropy, attention_probs
    
def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor, current_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate various metrics from logits and attention scores with causal masking."""
    logits_entropy, logits_varentropy = calculate_varentropy_logsoftmax(logits)
    attn_entropy, attn_varentropy, attention_probs = calculate_attention_varentropy(attention_scores, current_pos)
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))
    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": torch.mean(logits_entropy),
        "logits_varentropy": torch.mean(logits_varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }

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
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        new_entropy = -torch.sum(torch.where(probs > 0, probs * torch.log(probs), torch.tensor(0.0)), dim=-1)
        new_varentropy = torch.sum(probs * (torch.log(probs) + new_entropy.unsqueeze(-1))**2, dim=-1)

        # Update entropy and varentropy tensors
        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy

        return self