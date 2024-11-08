from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import json

#global imports
from entropix.local.config import SamplerState

def visualize_token_entropy_varentropy(metrics_data, generated_tokens, tokenizer, sampler_config):
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
        f"Token: {tokenizer.decode([token]) or 'Unknown'}<br>"
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
            (sampler_config.low_logits_entropy_threshold, 'rgba(255, 0, 0, 0.2)'),
            (sampler_config.medium_logits_entropy_threshold, 'rgba(0, 255, 0, 0.2)'),
            (sampler_config.high_logits_entropy_threshold, 'rgba(0, 0, 255, 0.2)')
        ], 'logits'),
        ('logits_varentropy', 'y', [
            (sampler_config.low_logits_varentropy_threshold, 'rgba(255, 165, 0, 0.2)'),
            (sampler_config.medium_logits_varentropy_threshold, 'rgba(165, 42, 42, 0.2)'),
            (sampler_config.high_logits_varentropy_threshold, 'rgba(128, 0, 128, 0.2)')
        ], 'logits'),
        ('attention_entropy', 'x', [
            (sampler_config.low_attention_entropy_threshold, 'rgba(255, 192, 203, 0.2)'),
            (sampler_config.medium_attention_entropy_threshold, 'rgba(0, 255, 255, 0.2)'),
            (sampler_config.high_attention_entropy_threshold, 'rgba(255, 255, 0, 0.2)')
        ], 'attention'),
        ('attention_varentropy', 'y', [
            (sampler_config.low_attention_varentropy_threshold, 'rgba(70, 130, 180, 0.2)'),
            (sampler_config.medium_attention_varentropy_threshold, 'rgba(244, 164, 96, 0.2)'),
            (sampler_config.high_attention_varentropy_threshold, 'rgba(50, 205, 50, 0.2)')
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
        "tokens": [tokenizer.decode([token]) for token in generated_tokens],
        "logits_entropy": metrics_data['logits_entropy'],
        "logits_varentropy": metrics_data['logits_varentropy'],
        "attention_entropy": metrics_data['attention_entropy'],
        "attention_varentropy": metrics_data['attention_varentropy'],
        "thresholds": {
            "logits_entropy": {
                "low": sampler_config.low_logits_entropy_threshold,
                "medium": sampler_config.medium_logits_entropy_threshold,
                "high": sampler_config.high_logits_entropy_threshold
            },
            "logits_varentropy": {
                "low": sampler_config.low_logits_varentropy_threshold,
                "medium": sampler_config.medium_logits_varentropy_threshold,
                "high": sampler_config.high_logits_varentropy_threshold
            },
            "attention_entropy": {
                "low": sampler_config.low_attention_entropy_threshold,
                "medium": sampler_config.medium_attention_entropy_threshold,
                "high": sampler_config.high_attention_entropy_threshold
            },
            "attention_varentropy": {
                "low": sampler_config.low_attention_varentropy_threshold,
                "medium": sampler_config.medium_attention_varentropy_threshold,
                "high": sampler_config.high_attention_varentropy_threshold
            }
        }
    }

    # Save the data to a file using the same timestamp
    data_filename = f"entropix/results/entropy_data_{timestamp}.json"
    with open(data_filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"Data exported to {data_filename}")

    return fig

def visualize_sampler_metrics(entropies, varentropies, sampler_states, generated_tokens, tokenizer):
    # Create a plotly figure with subplots
    fig = go.Figure()
    
    # Get token texts
    token_texts = [tokenizer.decode([token]) for token in generated_tokens]
    
    # Define colors for sampler states
    colors = {
        SamplerState.FLOWING: {'bg': '#ADD8E6', 'text': '#000000'},      # light blue
        SamplerState.TREADING: {'bg': '#90EE90', 'text': '#000000'},     # light green
        SamplerState.EXPLORING: {'bg': '#FF8C00', 'text': '#000000'},    # dark orange
        SamplerState.RESAMPLING: {'bg': '#FF69B4', 'text': '#000000'},   # hot pink
        SamplerState.ADAPTIVE: {'bg': '#800080', 'text': '#FFFFFF'}      # purple
    }
    
    # Create unified hover text
    hover_template = (
        "Step: %{x}<br>" +
        "Value: %{y}<br>" +
        "Token: %{customdata[0]}<br>" +
        "State: %{customdata[1]}"
    )

    
    # Add entropy trace
    fig.add_trace(go.Scatter(
        x=list(range(len(entropies))),
        y=entropies,
        name='Entropy',
        line=dict(color='blue'),
        yaxis='y1',
        customdata=list(zip(
            token_texts if token_texts else [''] * len(entropies),
            [state.value for state in sampler_states]
        )),
        hovertemplate=hover_template
    ))
    
    # Add varentropy trace
    fig.add_trace(go.Scatter(
        x=list(range(len(varentropies))),
        y=varentropies,
        name='Varentropy',
        line=dict(color='red'),
        yaxis='y1',
        customdata=list(zip(
            token_texts if token_texts else [''] * len(varentropies),
            [state.value for state in sampler_states]
        )),
        hovertemplate=hover_template
    ))
    
    # Create state indicators
    state_colors = [colors[state]['bg'] for state in sampler_states]
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
                color=color['bg'],
                size=10,
                symbol='square',
            ),
            name=state.value,
            showlegend=True,
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Entropy, Varentropy and Sampler States over Generation Steps',
            y=0.95,  # Move title down slightly
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title='Generation Step',
            showticklabels=True,
            tickmode='linear',
            dtick=5
        ),
        yaxis=dict(
            title='Value',
            domain=[0.25, 0.95]  
        ),
        yaxis2=dict(
            domain=[0.1, 0.2],  
            showticklabels=False,
            range=[-0.5, 0.5]
        ),
        height=750,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.0, 
            xanchor="right",
            x=1,
            orientation="h"
        ),
        margin=dict(t=100)
    )
    
    # Generate timestamp and save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"entropix/results/sampler_metrics_{timestamp}.html"

        # Create HTML content with the figure and text below
    html_content = f"""
    <html>
    <head>
        <style>
            .container {{
                display: flex;
                flex-direction: column;
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
            }}
            .plot-container {{
                width: 100%;
            }}
            .text-container {{
                margin-top: 20px;
                padding: 20px;
                border-top: 1px solid #ccc;
                max-height: 400px;
                overflow-y: auto;
                font-family: monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .token {{
                display: inline-block;
                padding: 2px 4px;
                margin: 2px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="plot-container">
                {fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            <div class="text-container">
                <h3>Generated Tokens:</h3>
                {''.join([f'<span class="token" style="background-color: {colors[state]["bg"]}; color: {colors[state]["text"]};">{token}</span>' 
                        for token, state in zip(token_texts, sampler_states)])}
            </div>
        </div>
    </body>
    </html>
    """
    # Write the complete HTML file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Sampler metrics visualization saved to {filename}")
    
    return fig


def visualize_logit_shift():
    pass