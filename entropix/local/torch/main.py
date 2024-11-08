from typing import Generator
import os
from dotenv import load_dotenv
import jax
from datetime import datetime
import torch
import torch.nn.functional as F
from pathlib import Path
import tyro

import pandas as pd
import csv

# global inports
from entropix.local.torch_weights import download_weights_torch, load_weights_torch
from entropix.local.config import (
    EntropixConfig,
    SamplerConfig,
    MODEL_IDS,
    MODEL_CONFIGS,
    get_model_params,
)
from entropix.local.tokenizer import download_tokenizer, Tokenizer

# framework specific imports
from entropix.local.torch.utils import precompute_freqs_cis, build_attn_mask, validate_csv
from entropix.local.torch.kvcache import KVCache
from entropix.local.torch.model import xfmr
from entropix.local.torch.sampler import sample
from entropix.local.torch.dslider import adaptive_dirichlet_step, initialize_state
from entropix.local.torch.dslider_config import DEFAULT_DS_CONFIG
from entropix.local.torch.utils import generate_chat_prompt

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
if device == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")


class EntropixModel:
    def __init__(self, model_size: str = "1B", seed: int = 1337):
        """
        Initialize EntropixModel with specified model size.

        Args:
            model_size: One of "1B", or "3B"
        """
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model size. Choose from: {list(MODEL_CONFIGS.keys())}")

        self.model_size = model_size
        self.config = MODEL_CONFIGS[model_size]
        self.model_params = get_model_params(self.config)
        self.xfmr_weights = load_weights_torch(model_id=model_size)
        self.tokenizer = Tokenizer("entropix/data/tokenizer.model")
        self.sampler_config = SamplerConfig(model_size)
        self.entropix_config = EntropixConfig()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.max_length = self.model_params.max_seq_len
        # TODO: self.sample_fn

    def generate(self, tokens, return_as_tokens=False) -> str or List[int]:
        generated_tokens = []

        with torch.inference_mode():
            bsz, seqlen = tokens.shape
            cur_pos = 0
            attn_mask = build_attn_mask(seqlen, cur_pos)
            freqs_cis = precompute_freqs_cis(
                self.model_params.head_dim,
                self.model_params.max_seq_len,
                self.model_params.rope_theta,
                self.model_params.use_scaled_rope,
            )
            kvcache = KVCache.new(
                self.model_params.n_layers,
                bsz,
                self.model_params.max_seq_len,
                self.model_params.n_local_kv_heads,
                self.model_params.head_dim,
            ).to(device)
            cfg = DEFAULT_DS_CONFIG

            logits, kvcache, scores, _ = xfmr(
                self.xfmr_weights, self.model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask
            )
            state = initialize_state(logits=logits, bsz=bsz, config=cfg).to(device)
            # next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
            state, next_token, *_ = adaptive_dirichlet_step(
                key=self.generator, state=state, logits=logits[:, -1], config=cfg
            )

            gen_tokens = next_token
            output = self.tokenizer.decode([next_token.item()])
            generated_tokens.append(next_token.item())
            cur_pos = seqlen
            stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)

            while cur_pos < self.max_length:
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(
                    self.xfmr_weights,
                    self.model_params,
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos : cur_pos + 1],
                    kvcache,
                )
                # next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
                state, next_token, *_ = adaptive_dirichlet_step(
                    key=self.generator, state=state, logits=logits[:, -1], config=cfg
                )
                # print("\n" + next_token, next_token.shape)
                generated_tokens.append(next_token.item())
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                output += self.tokenizer.decode(next_token.tolist()[0])
                if torch.isin(next_token, stop).any():
                    break
        if return_as_tokens:
            return generated_tokens
        else:
            return output

    def generate_stream(self, tokens) -> Generator[str, None, None]:
        generated_tokens = []
        with torch.inference_mode():
            bsz, seqlen = tokens.shape
            cur_pos = 0
            attn_mask = build_attn_mask(seqlen, cur_pos)
            freqs_cis = precompute_freqs_cis(
                self.model_params.head_dim,
                self.model_params.max_seq_len,
                self.model_params.rope_theta,
                self.model_params.use_scaled_rope,
            )
            kvcache = KVCache.new(
                self.model_params.n_layers,
                bsz,
                self.model_params.max_seq_len,
                self.model_params.n_local_kv_heads,
                self.model_params.head_dim,
            ).to(device)
            cfg = DEFAULT_DS_CONFIG.to(device)

            # Generate first token
            logits, kvcache, scores, _ = xfmr(
                self.xfmr_weights, self.model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask
            )
            state = initialize_state(logits=logits, bsz=bsz, config=cfg).to(device)
            # next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
            state, next_token, *_ = adaptive_dirichlet_step(
                key=self.generator, state=state, logits=logits[:, -1], config=cfg
            )

            generated_tokens.append(next_token.item())
            # Yield first token
            token_text = self.tokenizer.decode([next_token.item()])
            yield token_text

            gen_tokens = next_token
            cur_pos = seqlen
            stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)

            # Generate remaining tokens
            while cur_pos < self.max_length:
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(
                    self.xfmr_weights,
                    self.model_params,
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos : cur_pos + 1],
                    kvcache,
                )
                # next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
                state, next_token, *_ = adaptive_dirichlet_step(
                    key=self.generator, state=state, logits=logits[:, -1], config=cfg
                )
                # print("\n\n Gen", next_token, next_token.shape)
                generated_tokens.append(next_token.item())

                # Update state and yield token
                gen_tokens = torch.cat((gen_tokens, next_token), dim=0)
                token_text = self.tokenizer.decode(next_token.tolist())
                yield token_text

                if torch.isin(next_token, stop).any():
                    break

    def tokenize(self, prompt: str, chat_template: bool = False) -> torch.Tensor:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Think step by step and show your reasoning before you answer.",
            },
            {"role": "user", "content": prompt},
        ]
        if chat_template:
            prompt = generate_chat_prompt(messages)
        print(prompt)
        tokens = self.tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
        return torch.tensor([tokens], dtype=torch.long).to(device)


def initialize_model(model_size: str = "1B"):
    download_weights_torch(model_size)
    _ = download_tokenizer()
    jax.clear_caches()
    torch.cuda.empty_cache()
    global entropix_model
    entropix_model = EntropixModel()
    print(f"Model initialized and ready to use! Model size: {model_size}")


def generate_text(prompt: str, stream: bool = True) -> str:
    global entropix_model
    if "entropix_model" not in globals():
        print("Model not initialized. Please run initialize_model() first.")
        return

    tokens = entropix_model.tokenize(prompt, chat_template=True)
    if stream:
        response = ""
        for token in entropix_model.generate_stream(tokens):
            print(token, end="", flush=True)
            response += token
        print()
    else:
        response = entropix_model.generate(tokens)
        print(response)


if __name__ == "__main__":
    seed = 1337
    torch.manual_seed(seed=seed)
    # python3 -m entropix.local.torch.main --prompt "Which number is larger 9.11 or 9.9?"
    # python3 -m entropix.local.torch.main --prompt "Which number is larger 9.11 or 9.9?" --stream
    torch.cuda.empty_cache()
    initialize_model()
    tyro.cli(generate_text)
