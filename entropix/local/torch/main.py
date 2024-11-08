from typing import NamedTuple, Optional, Tuple
import os
from dotenv import load_dotenv

import plotly.graph_objects as go
import numpy as np
import jax
import json
from datetime import datetime

import torch
import torch.nn.functional as F

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
from entropix.local.torch.torch_weights import download_weights_torch, load_weights_torch
from entropix.local.tokenizer import download_tokenizer, Tokenizer
from entropix.local.config import EntropixConfig, SamplerConfig, SamplerState, GenerateConfig, MODEL_CONFIGS, get_model_params

#framework specific imports
from entropix.local.torch.utils import precompute_freqs_cis, build_attn_mask, validate_csv
from entropix.local.torch.kvcache import KVCache
from entropix.local.torch.model import xfmr
from entropix.local.torch.sampler import sample
from entropix.local.torch.metrics import calculate_metrics
from entropix.local.torch.dslider import adaptive_dirichlet_step
from entropix.local.torch.dslider import initialize_state
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
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')



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
        self.xfmr_weights = load_weights_torch(model_id=model_size)
        self.tokenizer = Tokenizer('entropix/data/tokenizer.model')
        self.sampler_config = SamplerConfig(model_size)
        self.entropix_config = EntropixConfig()
        self.generator = torch.Generator(device=device).manual_seed(1337)

    def visualize_token_entropy_varentropy(self, metrics_data, generated_tokens):
        # Add check at the start of the method
        if not generated_tokens:
            print("No tokens generated yet - skipping visualization")
            return None
        
        # Extract data
        entropies = np.array(metrics_data['logits_entropy'])
        varentropies = np.array(metrics_data['logits_varentropy'])
        attention_entropies = np.array(metrics_data['attention_entropy'])
        attention_varentropies = np.array(metrics_data['attention_varentropy'])
  
        # Ensure all arrays have the same length
        min_length = min(len(entropies), len(varentropies), len(attention_entropies), len(attention_varentropies), len(generated_tokens))
        entropies = entropies[:min_length]

        varentropies = varentropies[:min_length]
        attention_entropies = attention_entropies[:min_length]
        attention_varentropies = attention_varentropies[:min_length]
        generated_tokens = generated_tokens[:min_length]

        positions = np.arange(min_length)

        # Create hover text
        hover_text = [
            f"Token: {self.tokenizer.decode([token]) or 'Unknown'}<br>"
            f"Position: {i}<br>"
            f"Logits Entropy: {entropies[i]:.4f}<br>"
            f"Logits Varentropy: {varentropies[i]:.4f}<br>"
            f"Attention Entropy: {attention_entropies[i]:.4f}<br>"
            f"Attention Varentropy: {attention_varentropies[i]:.4f}"
            for i, token in enumerate(generated_tokens)
        ]

        # Create the 3D scatter plot
        fig = go.Figure()

        # Add logits entropy/varentropy scatter
        fig.add_trace(go.Scatter3d(
            x=entropies,
            y=varentropies,
            z=positions,
            mode='markers',
            marker=dict(
                size=5,
                color=entropies,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Logits Entropy", x=0.85),
            ),
            text=hover_text,
            hoverinfo='text',
            name='Logits Entropy/Varentropy'
        ))

        # Add attention entropy/varentropy scatter
        fig.add_trace(go.Scatter3d(
            x=attention_entropies,
            y=attention_varentropies,
            z=positions,
            mode='markers',
            marker=dict(
                size=5,
                color=attention_entropies,
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title="Attention Entropy", x=1.0),
            ),
            text=hover_text,
            hoverinfo='text',
            name='Attention Entropy/Varentropy'
        ))

        # Calculate the limits for x, y, and z

        logits_x_min, logits_x_max = min(entropies), max(entropies)
        logits_y_min, logits_y_max = min(varentropies), max(varentropies)
        attention_x_min, attention_x_max = min(attention_entropies), max(attention_entropies)
        attention_y_min, attention_y_max = min(attention_varentropies), max(attention_varentropies)
        z_min, z_max = min(positions), max(positions)

        # Function to create threshold planes
        def create_threshold_plane(threshold, axis, color, name, data_type):
            if data_type == 'logits':
                x_min, x_max = logits_x_min, logits_x_max
                y_min, y_max = logits_y_min, logits_y_max
            else:  # attention
                x_min, x_max = attention_x_min, attention_x_max
                y_min, y_max = attention_y_min, attention_y_max

            if axis == 'x':
                return go.Surface(
                    x=[[threshold, threshold], [threshold, threshold]],
                    y=[[y_min, y_max], [y_min, y_max]],
                    z=[[z_min, z_min], [z_max, z_max]],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=name,
                    visible=False
                )
            elif axis == 'y':
                return go.Surface(
                    x=[[x_min, x_max], [x_min, x_max]],
                    y=[[threshold, threshold], [threshold, threshold]],
                    z=[[z_min, z_min], [z_max, z_max]],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=name,
                    visible=False
                )

        # Add threshold planes
        thresholds = [
            ('logits_entropy', 'x', [
                (self.sampler_config.low_logits_entropy_threshold, 'rgba(255, 0, 0, 0.2)'),
                (self.sampler_config.medium_logits_entropy_threshold, 'rgba(0, 255, 0, 0.2)'),
                (self.sampler_config.high_logits_entropy_threshold, 'rgba(0, 0, 255, 0.2)')
            ], 'logits'),
            ('logits_varentropy', 'y', [
                (self.sampler_config.low_logits_varentropy_threshold, 'rgba(255, 165, 0, 0.2)'),
                (self.sampler_config.medium_logits_varentropy_threshold, 'rgba(165, 42, 42, 0.2)'),
                (self.sampler_config.high_logits_varentropy_threshold, 'rgba(128, 0, 128, 0.2)')
            ], 'logits'),
            ('attention_entropy', 'x', [
                (self.sampler_config.low_attention_entropy_threshold, 'rgba(255, 192, 203, 0.2)'),
                (self.sampler_config.medium_attention_entropy_threshold, 'rgba(0, 255, 255, 0.2)'),
                (self.sampler_config.high_attention_entropy_threshold, 'rgba(255, 255, 0, 0.2)')
            ], 'attention'),
            ('attention_varentropy', 'y', [
                (self.sampler_config.low_attention_varentropy_threshold, 'rgba(70, 130, 180, 0.2)'),
                (self.sampler_config.medium_attention_varentropy_threshold, 'rgba(244, 164, 96, 0.2)'),
                (self.sampler_config.high_attention_varentropy_threshold, 'rgba(50, 205, 50, 0.2)')
            ], 'attention')
        ]

        for threshold_type, axis, threshold_list, data_type in thresholds:
            for threshold, color in threshold_list:
                fig.add_trace(create_threshold_plane(threshold, axis, color, f'{threshold_type.replace("_", " ").title()} Threshold: {threshold}', data_type))

        # Create buttons for toggling views
        buttons = [
            dict(
                label='Show All',
                method='update',
                args=[{'visible': [True] * len(fig.data)}]
            ),
            dict(
                label='Hide All',
                method='update',
                args=[{'visible': [True, True] + [False] * (len(fig.data) - 2)}]
            ),
            dict(
                label='Logits Only',
                method='update',
                args=[{'visible': [True, False] + [True if i < 6 else False for i in range(len(fig.data) - 2)]}]
            ),
            dict(
                label='Attention Only',
                method='update',
                args=[{'visible': [False, True] + [True if i >= 6 else False for i in range(len(fig.data) - 2)]}]
            )
        ]

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Entropy',
                yaxis_title='Varentropy',
                zaxis_title='Token Position',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title='',
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.1,
                xanchor='left',
                yanchor='top',
                pad={"r": 10, "t": 10},
                showactive=True,
                buttons=buttons
            )],
            autosize=True,
            legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        )

        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the interactive plot as HTML
        interactive_filename = f"entropix/results/token_entropy_visualization_{timestamp}.html"
        fig.write_html(interactive_filename, include_plotlyjs=True, full_html=True)
        print(f"3D token entropy visualization saved to {interactive_filename}")

        # Export data to file
        export_data = {
            "tokens": [self.tokenizer.decode([token]) for token in generated_tokens],
            "logits_entropy": metrics_data['logits_entropy'],
            "logits_varentropy": metrics_data['logits_varentropy'],
            "attention_entropy": metrics_data['attention_entropy'],
            "attention_varentropy": metrics_data['attention_varentropy'],
            "thresholds": {
                "logits_entropy": {
                    "low": self.sampler_config.low_logits_entropy_threshold,
                    "medium": self.sampler_config.medium_logits_entropy_threshold,
                    "high": self.sampler_config.high_logits_entropy_threshold
                },
                "logits_varentropy": {
                    "low": self.sampler_config.low_logits_varentropy_threshold,
                    "medium": self.sampler_config.medium_logits_varentropy_threshold,
                    "high": self.sampler_config.high_logits_varentropy_threshold
                },
                "attention_entropy": {
                    "low": self.sampler_config.low_attention_entropy_threshold,
                    "medium": self.sampler_config.medium_attention_entropy_threshold,
                    "high": self.sampler_config.high_attention_entropy_threshold
                },
                "attention_varentropy": {
                    "low": self.sampler_config.low_attention_varentropy_threshold,
                    "medium": self.sampler_config.medium_attention_varentropy_threshold,
                    "high": self.sampler_config.high_attention_varentropy_threshold
                }
            }
        }

        # Save the data to a file using the same timestamp
        data_filename = f"entropix/results/entropy_data_{timestamp}.json"
        with open(data_filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Data exported to {data_filename}")

        return fig

    def generate(self, prompt, max_tokens=600, debug=True, batch: bool = False):
        # Initialize lists to store metrics
        metrics_data = {
            'logits_entropy': [],
            'logits_varentropy': [],
            'attention_entropy': [],
            'attention_varentropy': [],
            'kl_divergence': []
        }
        sampler_states = []
        generated_tokens = []

        with torch.inference_mode():
            # tokens = self.tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", bos=True, eos=False, allowed_special='all')
            tokens = self.tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
            tokens = torch.tensor([tokens], dtype=torch.long).to(device)
            bsz, seqlen = tokens.shape
            cur_pos = 0
            attn_mask = build_attn_mask(seqlen, cur_pos)
            freqs_cis = precompute_freqs_cis(self.model_params.head_dim, self.model_params.max_seq_len, self.model_params.rope_theta, self.model_params.use_scaled_rope)
            kvcache = KVCache.new(self.model_params.n_layers, bsz, self.model_params.max_seq_len, self.model_params.n_local_kv_heads, self.model_params.head_dim).to(device)
            state = initialize_state(bsz, 128256, DEFAULT_DS_CONFIG)
            cfg = DEFAULT_DS_CONFIG

            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
            # next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
            state, next_token, kl = adaptive_dirichlet_step(state, logits, cfg)
            metrics = calculate_metrics(logits, scores, seqlen)
            for key in metrics_data.keys():
                if key in metrics:
                    metrics_data[key].append(metrics[key].item())
            metrics_data['kl_divergence'].append(kl.item())
            # sampler_states.append(sampler_state)

            gen_tokens = next_token
            output = self.tokenizer.decode([next_token.item()])
            generated_tokens.append(next_token.item())
            cur_pos = seqlen
            stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)

            while cur_pos < max_tokens:
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
                # next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
                state, next_token, kl = adaptive_dirichlet_step(state, logits, cfg)
                metrics = calculate_metrics(logits, scores, cur_pos)
                for key in metrics_data.keys():
                    if key in metrics:
                        metrics_data[key].append(metrics[key].item())
                # sampler_states.append(sampler_state)
                metrics_data['attention_entropy'].append(metrics['attn_entropy'].item())
                metrics_data['attention_varentropy'].append(metrics['attn_varentropy'].item())
                metrics_data['kl_divergence'].append(kl.item())
                generated_tokens.append(next_token.item())
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                output += self.tokenizer.decode(next_token.tolist()[0])
                if torch.isin(next_token, stop).any():
                    break

        if debug:
            #self.debug_visualize_metrics(metrics_data)
            self.visualize_sampler_metrics(metrics_data['logits_entropy'], metrics_data['logits_varentropy'], metrics_data['kl_divergence'], sampler_states, generated_tokens)
            fig = self.visualize_token_entropy_varentropy(metrics_data, generated_tokens)
            self.visualize_kl_divergence(metrics_data['kl_divergence'], generated_tokens)
            if not batch:
                fig.show()
        return output

    def visualize_kl_divergence(self, kl_divergence, generated_tokens):
        # Get token texts
        token_texts = [self.tokenizer.decode([token]) for token in generated_tokens]
        steps = list(range(len(kl_divergence)))

        # Create hover text
        hover_text = [
            f"Token: {token_texts[i]}<br>"
            f"Step: {i}<br>"
            f"KL Divergence: {kl_divergence[i]:.4f}"
            for i in range(len(kl_divergence))
        ]

        # Create the figure
        fig = go.Figure()

        # Add KL divergence trace
        fig.add_trace(go.Scatter(
            x=steps,
            y=kl_divergence,
            mode='lines+markers',
            marker=dict(size=6),
            text=hover_text,
            hoverinfo='text',
            name='KL Divergence'
        ))

        # Update layout
        fig.update_layout(
            title='KL Divergence over Generation Steps',
            xaxis_title='Generation Step',
            yaxis_title='KL Divergence',
            hovermode='closest'
        )

        # Generate timestamp and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entropix/results/kl_divergence_{timestamp}.html"
        fig.write_html(filename, include_plotlyjs=True, full_html=True)
        print(f"KL divergence visualization saved to {filename}")

        return fig

    def debug_visualize_metrics(self, metrics_data):
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Debug Visualization of Sampler Metrics', fontsize=16)

        for idx, (key, values) in enumerate(metrics_data.items()):
            if values:  # Only plot if we have data for this metric
                row = idx // 2
                col = idx % 2
                axs[row, col].plot(values)
                axs[row, col].set_title(key)
                axs[row, col].set_xlabel('Generation Step')
                axs[row, col].set_ylabel('Value')
                axs[row, col].grid(True)

        # Add entropy_attention visualization if we have both metrics
        if metrics_data['logits_entropy'] and metrics_data['attention_entropy']:
            axs[2, 0].scatter(metrics_data['logits_entropy'], metrics_data['attention_entropy'])
            axs[2, 0].set_title('entropy_attention')
            axs[2, 0].set_xlabel('Logits Entropy')
            axs[2, 0].set_ylabel('Attention Entropy')
            axs[2, 0].grid(True)

        # Add entropy_interaction_strength visualization if we have both metrics
        if metrics_data['logits_entropy'] and metrics_data['interaction_strength']:
            axs[2, 1].scatter(metrics_data['logits_entropy'], metrics_data['interaction_strength'])
            axs[2, 1].set_title('entropy_interaction_strength')
            axs[2, 1].set_xlabel('Logits Entropy')
            axs[2, 1].set_ylabel('Interaction Strength')
            axs[2, 1].grid(True)

        plt.tight_layout()
        #plt.show()

    def visualize_sampler_metrics(self, entropies, varentropies, kl_divergences, sampler_states, generated_tokens):
        # Create a plotly figure with subplots
        fig = go.Figure()
        
        # Get token texts
        token_texts = [self.tokenizer.decode([token]) for token in generated_tokens]
        
        # Define colors for sampler states
        colors = {
            SamplerState.FLOWING: 'lightblue',
            SamplerState.TREADING: 'lightgreen',
            SamplerState.EXPLORING: 'orange',
            SamplerState.RESAMPLING: 'pink',
            SamplerState.ADAPTIVE: 'purple'
        }
        
        # Create unified hover text
        hover_template = (
            "Step: %{x}<br>" +
            "Value: %{y}<br>" +
            "Token: %{customdata[0]}<br>" +
            "State: %{customdata[1]}"
        )
        
        # Scale KL divergences for better visualization
        scaled_kl_divergences = [kl/500 if kl is not None else None for kl in kl_divergences]
        
        # Create x-axis values with fewer ticks
        x_values = list(range(len(entropies)))
        tick_spacing = max(1, len(x_values) // 20)  # Show ~20 ticks on x-axis
        
        # Add entropy trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=entropies,
            name='Entropy',
            line=dict(color='blue'),
            yaxis='y1',
            customdata=list(zip(
                token_texts if token_texts else [''] * len(entropies),
                ['N/A'] * len(entropies) if not sampler_states else [state.value for state in sampler_states]
            )),
            hovertemplate=hover_template
        ))
        
        # Add varentropy trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=varentropies,
            name='Varentropy',
            line=dict(color='red'),
            yaxis='y1',
            customdata=list(zip(
                token_texts if token_texts else [''] * len(varentropies),
                ['N/A'] * len(varentropies) if not sampler_states else [state.value for state in sampler_states]
            )),
            hovertemplate=hover_template
        ))
        
        # Add KL divergence trace (scaled)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=scaled_kl_divergences,
            name='KL Divergence (/500)',
            line=dict(color='green'),
            yaxis='y1',
            customdata=list(zip(
                token_texts if token_texts else [''] * len(kl_divergences),
                ['N/A'] * len(kl_divergences) if not sampler_states else [state.value for state in sampler_states],
                kl_divergences  # Add original KL values to customdata
            )),
            hovertemplate=(
                "Step: %{x}<br>" +
                "Scaled Value: %{y:.4f}<br>" +
                "Actual Value: %{customdata[2]:.1f}<br>" +
                "Token: %{customdata[0]}<br>" +
                "State: %{customdata[1]}"
            )
        ))
        
        # Only add state indicators and legend if sampler_states is not empty
        if sampler_states:
            # Create state indicators
            state_colors = [colors[state] for state in sampler_states]
            state_names = [state.value for state in sampler_states]
            
            # Add state indicators
            fig.add_trace(go.Scatter(
                x=list(range(len(sampler_states))),
                y=[0] * len(sampler_states),
                mode='markers',
                marker=dict(
                    color=state_colors,
                    size=20,
                    symbol='square',
                ),
                customdata=list(zip(
                    token_texts if token_texts else [''] * len(sampler_states),
                    state_names
                )),
                hovertemplate=hover_template,
                yaxis='y2',
                showlegend=False,
            ))
            
            # Add state legend
            for state, color in colors.items():
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=10,
                        symbol='square',
                    ),
                    name=state.value,
                    showlegend=True,
                ))
        
        # Update layout
        title_text = 'Entropy, Varentropy, KL Divergence'
        if sampler_states:
            title_text += ' and Sampler States'
        title_text += ' over Generation Steps'
        
        fig.update_layout(
            title=title_text,
            xaxis=dict(
                title='Generation Step',
                showticklabels=True,
                tickmode='array',
                tickvals=list(range(0, len(entropies), tick_spacing)),
                ticktext=list(range(0, len(entropies), tick_spacing))
            ),
            yaxis=dict(
                title='Value',
                domain=[0.25 if sampler_states else 0.15, 0.95]  
            ),
            height=750,
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                orientation="h"
            )
        )
        
        if sampler_states:
            fig.update_layout(
                yaxis2=dict(
                    domain=[0.1, 0.2],
                    showticklabels=False,
                    range=[-0.5, 0.5]
                )
            )
        
        # Add tokens with color based on either sampler states or KL divergence
        formatted_text = ""
        line_length = 0
        max_line_length = 180
        
        # Calculate KL divergence threshold for coloring (e.g., 75th percentile)
        if not sampler_states and kl_divergences:
            kl_threshold = np.percentile(kl_divergences, 75)
        
        for i, token in enumerate(token_texts):
            if sampler_states:
                # Use sampler state colors
                color = colors[sampler_states[i]]
            else:
                # Use KL divergence-based coloring
                if kl_divergences and i < len(kl_divergences):
                    # Interpolate color between black and red based on KL divergence
                    kl_value = kl_divergences[i]
                    if kl_value > kl_threshold:
                        # Calculate red intensity (0 to 255) based on how much the value exceeds the threshold
                        red_intensity = min(255, int(255 * (kl_value - kl_threshold) / kl_threshold))
                        color = f'rgb({red_intensity}, 0, 0)'
                    else:
                        color = 'black'
                else:
                    color = 'black'
            
            token_text = f"<span style='color: {color}'>{token}</span> "
            
            # Add newline if current line would be too long
            if line_length + len(token) > max_line_length:
                formatted_text += "<br>"
                line_length = 0
            
            formatted_text += token_text
            line_length += len(token) + 1  # +1 for the space
        
        # Add the text
        fig.add_annotation(
            text=formatted_text,
            xref="paper",
            yref="paper",
            x=0,
            y=0.07,
            showarrow=False,
            font=dict(size=20),
            align="left",
            xanchor="left",
            yanchor="top",
            xshift=5,
            yshift=0,
            bordercolor="gray",
            borderwidth=0,
        )
        
        # Calculate bottom margin based on text length
        num_lines = formatted_text.count('<br>') + 1
        bottom_margin = max(30, num_lines * 25)  # Increased multiplier for better spacing
        
        # Update layout to allow text to extend
        fig.update_layout(
            margin=dict(b=bottom_margin),
            height=max(750, 750 + bottom_margin - 30)  # Dynamically increase height based on text
        )
        
        # Generate timestamp and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entropix/results/sampler_metrics_{timestamp}.html"
        fig.write_html(filename, include_plotlyjs=True, full_html=True)
        print(f"Sampler metrics visualization saved to {filename}")
        
        return fig

    def generate_stream(self, prompt: str, max_tokens: int = 600, debug: bool = True, batch: bool = False) -> str:
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
            'attention_varentropy': [],
            'kl_divergence': []
        }
        sampler_states = []
        generated_tokens = []
        with torch.inference_mode():
            # tokens = self.tokenizer.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", bos=True, eos=False, allowed_special='all')
            tokens = self.tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
            tokens = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Initial setup
            bsz, seqlen = tokens.shape
            cur_pos = 0
            attn_mask = build_attn_mask(seqlen, cur_pos)
            freqs_cis = precompute_freqs_cis(self.model_params.head_dim, self.model_params.max_seq_len, self.model_params.rope_theta, self.model_params.use_scaled_rope)
            kvcache = KVCache.new(self.model_params.n_layers, bsz, self.model_params.max_seq_len, self.model_params.n_local_kv_heads, self.model_params.head_dim).to(device)
            cfg = DEFAULT_DS_CONFIG.to(device)
            state = initialize_state(bsz, 128256, cfg, device=device)

            # Generate first token
            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
            # next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
            state, next_token, kl = adaptive_dirichlet_step(state, logits[:, -1], cfg)
            # Track metrics
            metrics = calculate_metrics(logits, scores, seqlen)
            for key in metrics_data.keys():
                if key in metrics:
                    metrics_data[key].append(metrics[key].item())
            metrics_data['kl_divergence'].append(kl.item())
            # sampler_states.append(sampler_state)
            generated_tokens.append(next_token.item())

            # Yield first token
            token_text = self.tokenizer.decode([next_token.item()])
            yield token_text

            gen_tokens = next_token
            cur_pos = seqlen
            stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)

            # Generate remaining tokens
            while cur_pos < max_tokens:
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
                # next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
                state, next_token, kl = adaptive_dirichlet_step(state, logits[:, -1], cfg)

                # Track metrics
                metrics = calculate_metrics(logits, scores, cur_pos)
                for key in metrics_data.keys():
                    if key in metrics:
                        metrics_data[key].append(metrics[key].item())
                # sampler_states.append(sampler_state)
                metrics_data['attention_entropy'].append(metrics['attn_entropy'].item())
                metrics_data['attention_varentropy'].append(metrics['attn_varentropy'].item())
                metrics_data['kl_divergence'].append(kl.item())
                generated_tokens.append(next_token.item())

                # Update state and yield token
                gen_tokens = torch.cat((gen_tokens, next_token), dim=0)
                token_text = self.tokenizer.decode(next_token.tolist())
                yield token_text

                if torch.isin(next_token, stop).any():
                    break

        if debug and len(generated_tokens) > 0:  # Only show visualizations if we have data
            self.visualize_sampler_metrics(metrics_data['logits_entropy'], metrics_data['logits_varentropy'], metrics_data['kl_divergence'], sampler_states, generated_tokens)
            fig = self.visualize_token_entropy_varentropy(metrics_data, generated_tokens)
            self.visualize_kl_divergence(metrics_data['kl_divergence'], generated_tokens)
            # if not batch:
            #     fig.show()

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
    torch.cuda.empty_cache()

    global entropix_model

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

    response = ""
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
                prompt = generate_chat_prompt([{ "role": "user", "content": prompt }])
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
            prompt = generate_chat_prompt([{ "role": "user", "content": config.prompt }])
            for token in entropix_model.generate_stream(prompt, config.max_tokens, config.debug):
                print(token, end='', flush=True)
                response += token
            print()  # Final newline
        else:
            response = entropix_model.generate(config.prompt, config.max_tokens, config.debug)
            print(response)



if __name__ == '__main__':
    torch.cuda.empty_cache()
    initialize_model()
    tyro.cli(generate_text)
