import asyncio
import dataclasses
import functools
import itertools
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
from concurrent import futures
from typing import Any, List, Optional, Tuple, TypeVar, Generic

import torch
from pathlib import Path

from entropix.local.tokenizer import Tokenizer
from entropix.local.torch.kvcache import KVCache
from entropix.local.torch.model import xfmr
from entropix.local.torch.sampler import sample
from entropix.local.torch.utils import precompute_freqs_cis

V = TypeVar('V')

@dataclasses.dataclass 
class ReturnSample:
    text: List[str]
    token_ids: List[int]

@dataclasses.dataclass
class ActiveRequestMetadata:
    start_time: Optional[float] = None
    prefill_enqueue_time: Optional[float] = None
    prefill_dequeue_time: Optional[float] = None
    transfer_enqueue_time: Optional[float] = None
    transfer_dequeue_time: Optional[float] = None
    generate_enqueue_time: Optional[float] = None
    generate_dequeue_time: Optional[float] = None
    complete_time: Optional[float] = None

@dataclasses.dataclass
class ActiveRequest:
    max_tokens: int
    return_channel: "AsyncMultifuture[List[ReturnSample]]"
    complete: Optional[torch.Tensor] = None
    prefill_result: Any = None
    prefill_content: Optional[str | List[int]] = None
    generate_timestep_added: Optional[int] = None
    is_client_side_tokenization: Optional[bool] = False
    metadata: ActiveRequestMetadata = dataclasses.field(default_factory=ActiveRequestMetadata)

    def enqueue_samples(self, generated_samples: List[ReturnSample]):
        self.return_channel.add_result(generated_samples)

class AsyncMultifuture(Generic[V]):
    def __init__(self):
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()

    def cancel(self, unused: Any = None):
        self._cancelled.set()
        self.set_exception(futures.CancelledError())

    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def done(self) -> bool:
        return self._done.is_set()

    def set_exception(self, exception: Exception):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, _Exception(exception))
        self._loop.call_soon_threadsafe(self._done.set)

    def add_result(self, result: V):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, result)

    def close(self):
        self.set_exception(StopAsyncIteration())

    def __aiter__(self):
        return self

    async def __anext__(self) -> V:
        value = await self._queue.get()
        if isinstance(value, _Exception):
            raise value.exception
        return value

class EntropixOrchestrator:
    def __init__(self, xfmr_weights, model_params, tokenizer):
        self.xfmr_weights = xfmr_weights
        self.model_params = model_params
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def decode(self, request):
        return_channel = AsyncMultifuture()
        active_request = ActiveRequest(
            max_tokens=request.max_tokens,
            prefill_content=request.tokens,
            return_channel=return_channel,
            is_client_side_tokenization=request.is_client_side_tokenization,
            metadata=ActiveRequestMetadata(
                start_time=time.time(),
                prefill_enqueue_time=time.perf_counter()
            )
        )

        # Tokenize input
        if isinstance(active_request.prefill_content, str):
            tokens = self.tokenizer.encode(active_request.prefill_content, bos=False, eos=False)
            tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)
        else:
            tokens = torch.tensor([active_request.prefill_content], dtype=torch.long).to(self.device) 

        with torch.inference_mode():
            bsz, seqlen = tokens.shape
            freqs_cis = precompute_freqs_cis(
                self.model_params.head_dim,
                self.model_params.max_seq_len,
                self.model_params.rope_theta,
                self.model_params.use_scaled_rope
            ).to(self.device)

            kvcache = KVCache.new(
                self.model_params.n_layers,
                bsz,
                self.model_params.max_seq_len,
                self.model_params.n_local_kv_heads,
                self.model_params.head_dim
            ).to(self.device)

            # Initial token generation
            cur_pos = 0
            logits, kvcache, scores, _ = xfmr(
                self.xfmr_weights, 
                self.model_params,
                tokens,
                cur_pos,
                freqs_cis[:seqlen],
                kvcache
            )

            next_token = sample(logits[:, -1])
            token_text = self.tokenizer.decode([next_token.item()])
            yield [(token_text, [next_token.item()])]

            gen_tokens = next_token

            # Generate remaining tokens
            for i in range(request.max_tokens - 1):
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(
                    self.xfmr_weights,
                    self.model_params, 
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos:cur_pos+1],
                    kvcache
                )

                next_token = sample(logits[:, -1])
                token_text = self.tokenizer.decode([next_token.item()])
                yield [(token_text, [next_token.item()])]

                if next_token.item() in self.tokenizer.stop_tokens:
                    break

                gen_tokens = torch.cat((gen_tokens, next_token), dim=0)