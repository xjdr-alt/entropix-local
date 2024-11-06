import functools
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import numpy as np

from entropix.local.config import ModelParams
from entropix.local.tokenizer import Tokenizer
from entropix.local.torch.kvcache import KVCache
from entropix.local.torch.dslider import initialize_state
from entropix.local.torch.dslider_config import DEFAULT_DS_CONFIG

@dataclass
class LayerWeights:
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor

@dataclass 
class XfmrWeights:
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: List[LayerWeights]

class SlotData(NamedTuple):
    tokens: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor

class ResultTokens(NamedTuple):
    data: torch.Tensor
    tokens_idx: Tuple[int, int]
    valid_idx: Tuple[int, int] 
    length_idx: Tuple[int, int]
    samples_per_slot: int

    def copy_to_host_async(self):
        if self.data.device.type != 'cpu':
            self.data = self.data.cpu()

    def convert_to_numpy(self):
        return ResultTokens(
            self.data.cpu().numpy(),
            self.tokens_idx,
            self.valid_idx, 
            self.length_idx,
            self.samples_per_slot
        )

    def get_result_at_slot(self, slot: int) -> SlotData:
        start_idx = slot * self.samples_per_slot
        end_idx = (slot + 1) * self.samples_per_slot
        return SlotData(
            tokens=self.data[start_idx:end_idx, self.tokens_idx[0]:self.tokens_idx[1]],
            valid=self.data[start_idx:end_idx, self.valid_idx[0]:self.valid_idx[1]],
            lengths=self.data[start_idx:end_idx, self.length_idx[0]:self.length_idx[1]][:, 0]
        )

class EntropixEngine:
    def __init__(
        self,
        params: ModelParams,
        xfmr_weights: XfmrWeights,
        tokenizer: Tokenizer,
        xfmr_fn: Callable,
        sample_fn: Callable,
    ):
        self.params = params
        self.xfmr_weights = xfmr_weights
        self.tokenizer = tokenizer
        self.xfmr_fn = xfmr_fn
        self.sample_fn = sample_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.freqs_cis = self.precompute_freqs_cis(
            params.head_dim,
            params.max_seq_len,
            params.rope_theta,
            params.use_scaled_rope
        ).to(self.device)

    def get_tokenizer(self) -> Dict[str, Any]:
        return {}

    def build_tokenizer(self, metadata: Dict[str, Any]) -> Tokenizer:
        return self.tokenizer

    @property
    def max_concurrent_decodes(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 1

    @property
    def samples_per_slot(self) -> int:
        return 1

    @property
    def max_prefill_length(self) -> int:
        return 1024

    def free_resource(self, slot: int) -> None:
        pass

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        SCALE_FACTOR = 8
        LOW_FREQ_FACTOR = 1 
        HIGH_FREQ_FACTOR = 4
        OLD_CONTEXT_LEN = 8192

        low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
        high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR
        
        wavelen = 2 * math.pi / freqs
        scale = torch.ones_like(freqs)
        
        # High frequency case
        high_mask = wavelen < high_freq_wavelen
        scale[high_mask] = 1.0
        
        # Low frequency case  
        low_mask = wavelen > low_freq_wavelen
        scale[low_mask] = 1.0 / SCALE_FACTOR
        
        # Mid frequency case
        mid_mask = ~(high_mask | low_mask)
        if mid_mask.any():
            smooth = (OLD_CONTEXT_LEN / wavelen[mid_mask] - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
            scale[mid_mask] = (1 - smooth) / SCALE_FACTOR + smooth
            
        return freqs * scale

    def precompute_freqs_cis(
        self,
        dim: int,
        end: int,
        theta: float = 500000.0,
        use_scaled: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].to(dtype) / dim))
        if use_scaled:
            freqs = self.apply_scaling(freqs)
        t = torch.arange(end, dtype=dtype)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def build_attn_mask(self, seqlen: int, start_pos: int) -> torch.Tensor:
        mask = torch.zeros((seqlen, seqlen), dtype=torch.float32)
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            pad = torch.zeros((seqlen, start_pos), dtype=torch.float32)
            mask = torch.cat([pad, mask], dim=1)
        return mask

    @torch.inference_mode()
    def prefill(
        self,
        params: ModelParams,
        padded_tokens: torch.Tensor,
        true_length: int,
        top_k: int = 6,
    ) -> Tuple[Dict[str, Any], ResultTokens]:
        cur_pos = 0
        bsz, seqlen = padded_tokens.shape
        attn_mask = self.build_attn_mask(seqlen, cur_pos).to(self.device)
        
        kvcache = KVCache.new(
            params.n_layers,
            bsz, 
            params.max_seq_len,
            params.n_local_kv_heads,
            params.head_dim
        ).to(self.device)

        logits, kvcache, _ = self.xfmr_fn(
            self.xfmr_weights,
            params,
            padded_tokens,
            cur_pos,
            self.freqs_cis[:seqlen],
            kvcache,
            attn_mask=attn_mask
        )

        values, next_token = torch.topk(logits[:, true_length], k=top_k)
        next_token = next_token.reshape(top_k, 1)

        tokens = next_token
        validity = torch.ones_like(next_token, dtype=torch.bool)
        lengths = torch.full((tokens.shape[0], 1), true_length + 1, dtype=torch.int32)
        
        data = torch.cat([tokens, validity, lengths], dim=1)
        
        result = ResultTokens(
            data=data,
            tokens_idx=(0, 1),
            valid_idx=(1, 2), 
            length_idx=(2, 3),
            samples_per_slot=bsz
        )

        prefix = {
            "logits": logits,
            "cache": kvcache,
            "next_pos": true_length + 1,
            "generated_tokens": torch.zeros((bsz, 1), dtype=torch.int32),
            "tokens": next_token
        }

        return prefix, result

    @torch.inference_mode()
    def generate(
        self,
        params: ModelParams,
        decode_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], ResultTokens]:
        cur_pos = decode_state["next_pos"]
        bsz = decode_state["tokens"].shape[0]
        freqs_cis_slice = self.freqs_cis[cur_pos:cur_pos+1]

        logits, kvcache, _ = self.xfmr_fn(
            self.xfmr_weights,
            params,
            decode_state["tokens"],
            cur_pos,
            freqs_cis_slice,
            decode_state["cache"]
        )

        new_state, new_token = self.sample_fn(
            decode_state["dslider_state"], 
            logits[:, -1, :],
            DEFAULT_DS_CONFIG
        )
        new_token = new_token.reshape(bsz, 1)

        result = ResultTokens(
            data=torch.cat([
                new_token,
                torch.ones_like(new_token, dtype=torch.bool),
                torch.full((bsz, 1), decode_state["generated_tokens"][:, -1] + 1, dtype=torch.int32)
            ], dim=1),
            tokens_idx=(0, 1),
            valid_idx=(1, 2),
            length_idx=(2, 3),
            samples_per_slot=bsz
        )

        new_decode_state = {
            "logits": logits,
            "cache": kvcache,
            "next_pos": decode_state["next_pos"] + 1,
            "generated_tokens": decode_state["generated_tokens"] + 1,
            "tokens": new_token,
            "dslider_state": new_state
        }

        return new_decode_state, result

    def insert(
        self,
        prefix: Dict[str, Any],
        decode_state: Dict[str, Any],
        slot: int,
    ) -> Dict[str, Any]:
        bsz = prefix["tokens"].shape[0]
        layers, _, max_seq_len, kv_heads, head_dim = prefix["cache"].k.shape
        
        new_k = prefix["cache"].k.expand(layers, bsz, max_seq_len, kv_heads, head_dim)
        new_v = prefix["cache"].v.expand(layers, bsz, max_seq_len, kv_heads, head_dim)
        new_cache = KVCache(k=new_k, v=new_v)

        return {
            "logits": prefix["logits"],
            "cache": new_cache, 
            "next_pos": prefix["next_pos"],
            "generated_tokens": prefix["generated_tokens"],
            "tokens": prefix["tokens"],
            "dslider_state": initialize_state(
                bsz,
                prefix["logits"].shape[-1],
                DEFAULT_DS_CONFIG,
                device=self.device
            )
        }