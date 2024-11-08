from typing import NamedTuple, Optional, Tuple
import os
from dotenv import load_dotenv

import plotly.graph_objects as go
import numpy as np
import jax
import json
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn

import math

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from pathlib import Path
from functools import partial
import tyro

import pandas as pd
import csv

#global imports
from entropix.local.torch.torch_weights import download_weights_torch
from entropix.local.mlx.mlx_weights import load_weights_mx
from entropix.local.tokenizer import download_tokenizer, Tokenizer
from entropix.local.config import EntropixConfig, SamplerConfig, SamplerState, GenerateConfig, MODEL_CONFIGS, get_model_params
from entropix.local.visuals import visualize_sampler_metrics, visualize_token_entropy_varentropy, visualize_logit_shift

#framework specific imports
from entropix.local.mlx.utils import precompute_freqs_cis, build_attn_mask, validate_csv
from entropix.local.mlx.kvcache import KVCache
from entropix.local.mlx.model import xfmr
from entropix.local.mlx.sampler import sample
from entropix.local.mlx.metrics import calculate_metrics


#print(f"Using device: {device}")

class EntropixModel:
    def __init__(self, model_size: str = "1B"):
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
        self.xfmr_weights = load_weights_mx(model_id=model_size)
        self.tokenizer = Tokenizer('entropix/data/tokenizer.model')
        self.sampler_config = SamplerConfig(model_size)
        self.entropix_config = EntropixConfig()
        self.rng_key =  mx.random.key(1337)

 
    def generate(self, prompt, max_tokens=600, debug=True, batch: bool = False):
        # Initialize lists to store metrics
        metrics_data = {
            'logits_entropy': [],
            'logits_varentropy': [],
            'attention_entropy': [],
            'attention_varentropy': []
        }
        sampler_states = []
        generated_tokens = []


        tokens = self.tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", bos=True, eos=False, allowed_special='all')
        tokens = mx.array([tokens], dtype=mx.int32)
        bsz, seqlen = tokens.shape
        cur_pos = 0
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(self.model_params.head_dim, self.model_params.max_seq_len, self.model_params.rope_theta, self.model_params.use_scaled_rope)
        kvcache = KVCache.new(self.model_params.n_layers, bsz, self.model_params.max_seq_len, self.model_params.n_local_kv_heads, self.model_params.head_dim)

        logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, seqlen, rng_key=self.rng_key)

        metrics = calculate_metrics(logits, scores, seqlen)
        for key in metrics_data.keys():
            if key in metrics:
                metrics_data[key].append(metrics[key].item())
        sampler_states.append(sampler_state)

        gen_tokens = next_token
        output = self.tokenizer.decode([next_token.item()])
        generated_tokens.append(next_token.item())
        cur_pos = seqlen
        stop = mx.array([128001, 128008, 128009], dtype=mx.int32) 

        while cur_pos < max_tokens:
            cur_pos += 1
            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
            next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, cur_pos, rng_key=self.rng_key)

            metrics = calculate_metrics(logits, scores, cur_pos)
            for key in metrics_data.keys():
                if key in metrics:
                    metrics_data[key].append(metrics[key].item())
            sampler_states.append(sampler_state)
            metrics_data['attention_entropy'].append(metrics['attn_entropy'].item())
            metrics_data['attention_varentropy'].append(metrics['attn_varentropy'].item())
            generated_tokens.append(next_token.item())

            gen_tokens = mx.concatenate((gen_tokens, next_token), axis=1)
            output += self.tokenizer.decode(next_token.tolist()[0])
            if mx.any(mx.concatenate([next_token == stop_token for stop_token in stop])):
                break

        if debug:
            visualize_sampler_metrics(metrics_data['logits_entropy'], metrics_data['logits_varentropy'], sampler_states, generated_tokens, self.tokenizer)
            visualize_token_entropy_varentropy(metrics_data, generated_tokens, self.tokenizer, self.sampler_config)

        return output
    
    def generate_stream(self, prompt: str, max_tokens: int = 4000, debug: bool = True, batch: bool = False) -> str:
        """Stream tokens as they're generated.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            debug: Whether to show debug visualizations
        
        Yields:
            Generated tokens as strings
        """
        # Initialize metrics tracking
        metrics_data = {
            'logits_entropy': [],
            'logits_varentropy': [],
            'attention_entropy': [],
            'attention_varentropy': []
        }
        sampler_states = []
        generated_tokens = []


        tokens = self.tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", bos=True, eos=False, allowed_special='all')
        tokens = mx.array([tokens], dtype=mx.int32)
        
        # Initial setup
        bsz, seqlen = tokens.shape
        cur_pos = 0
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(self.model_params.head_dim, self.model_params.max_seq_len, self.model_params.rope_theta, self.model_params.use_scaled_rope)
        kvcache = KVCache.new(self.model_params.n_layers, bsz, self.model_params.max_seq_len, self.model_params.n_local_kv_heads, self.model_params.head_dim)

        # Generate first token
        logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, seqlen, rng_key=self.rng_key)

        # Track metrics
        metrics = calculate_metrics(logits, scores, seqlen)
        for key in metrics_data.keys():
            if key in metrics:
                metrics_data[key].append(metrics[key].item())
        sampler_states.append(sampler_state)
        generated_tokens.append(next_token.item())

        # Yield first token
        token_text = self.tokenizer.decode([next_token.item()])
        yield token_text

        gen_tokens = next_token
        cur_pos = seqlen
        stop = mx.array([128001, 128008, 128009], dtype=mx.int32)

        # Generate remaining tokens
        while cur_pos < max_tokens:
            cur_pos += 1
            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
            next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, cur_pos, rng_key=self.rng_key)

            # Track metrics
            metrics = calculate_metrics(logits, scores, cur_pos)
            for key in metrics_data.keys():
                if key in metrics:
                    metrics_data[key].append(metrics[key].item())
            sampler_states.append(sampler_state)
            metrics_data['attention_entropy'].append(metrics['attn_entropy'].item())
            metrics_data['attention_varentropy'].append(metrics['attn_varentropy'].item())
            generated_tokens.append(next_token.item())

            # Update state and yield token
            gen_tokens = mx.concatenate((gen_tokens, next_token), axis=1)
            token_text = self.tokenizer.decode(next_token.tolist()[0])
            yield token_text

            if mx.any(mx.concatenate([next_token == stop_token for stop_token in stop])):
                break

        if debug and len(generated_tokens) > 0:  # Only show visualizations if we have data
            visualize_sampler_metrics(metrics_data['logits_entropy'], metrics_data['logits_varentropy'], sampler_states, generated_tokens, self.tokenizer)
            visualize_token_entropy_varentropy(metrics_data, generated_tokens, self.tokenizer, self.sampler_config)

# Function to initialize the model (to be run once)
def initialize_model() -> None:
    #create the library structure
    folders = [
        "entropix/data",
        "entropix/results",
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    load_dotenv(override=True)
    model_size = os.getenv('SELECTED_MODEL_SIZE', '1B')
    download_weights_torch(model_size)
    _ = download_tokenizer()
    jax.clear_caches()

    global entropix_model
    print("You are running the MLX model. This only runs on Apple Silicon.")

    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model size. Choose from: {list(MODEL_CONFIGS.keys())}")
    print(f"Initializing {model_size} model...")
    entropix_model = EntropixModel(model_size)
    print(f"{model_size} model initialized and ready to use!")

# Function to generate text (can be used in multiple cells)
def generate_text(config: GenerateConfig) -> None:
    """Generate text using the model with the given configuration.
    
    Args:
        config: Generation configuration parameters
    """
    global entropix_model
    if 'entropix_model' not in globals():
        print("Model not initialized. Please run initialize_model() first.")
        return

    # Handle CSV input if provided
    if config.csv_file:
        csv_path = "entropix/prompts/" + config.csv_file
        if not validate_csv(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        total_prompts = len(df)
        
        print(f"Processing {total_prompts} prompts from CSV file...")
        
        # Create output CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"entropix/results/generated_responses_{timestamp}.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['prompts', 'response'])
            
            for idx, row in df.iterrows():
                prompt = row['prompts'].strip()
                print(f"\nProcessing prompt {idx + 1}/{total_prompts}:")
                print(f"Prompt: {prompt}\n")
                
                
                if config.stream:
                    response = ""
                    print("Response: ", end='', flush=True)
                    for token in entropix_model.generate_stream(prompt, config.max_tokens, config.debug, batch=True):
                        print(token, end='', flush=True)
                        response += token
                    print()  # Final newline
                else:
                    response = entropix_model.generate(prompt, config.max_tokens, config.debug, batch=True)
                    print(f"Response: {response}\n")
                
                writer.writerow([prompt, response])
                
                
        print(f"\nAll responses have been saved to {output_file}")
        
    else:
        # Original single prompt behavior
        if config.stream:
            response = ""
            for token in entropix_model.generate_stream(config.prompt, config.max_tokens, config.debug):
                print(token, end='', flush=True)
                response += token
            print()  # Final newline
        else:
            response = entropix_model.generate(config.prompt, config.max_tokens, config.debug)
            print(response)



if __name__ == '__main__':
    initialize_model()
    tyro.cli(generate_text)