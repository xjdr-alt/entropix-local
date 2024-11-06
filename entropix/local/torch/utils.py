import torch
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from typing import List

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 2048  # original llama3 length

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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
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
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask

def validate_csv(file_path: str) -> bool:
    """Validate if the CSV file exists and contains a 'prompt' column with string data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if file exists and is a CSV
        if not file_path.endswith('.csv'):
            print("Error: File must be a CSV file")
            return False
            
        # Try reading the CSV
        df = pd.read_csv(file_path)
        
        # Check if 'prompts' column exists
        if 'prompts' not in df.columns:
            print("Error: CSV must contain a 'prompts' column")
            return False
            
        # Check if prompts are strings and not empty
        if not all(isinstance(x, str) and len(x.strip()) > 0 for x in df['prompts'].dropna()):
            print("Error: All prompts must be non-empty strings")
            return False
            
        return True
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return False
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        return False
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return False
    
@dataclass
class Message:
    def __init__(self, content, role):
        self.content = content
        self.role = role

chat_preamble = "You are a helpful assistant."
role_to_header = {
    "system": "user",
    "user": "assistant",
    "assistant": "user",
}
def formatted_date():
    current_date = datetime.now()
    return current_date.strftime("%d %B %Y")

def system_header() -> str:
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"

def env(tools: bool) -> str:
    if tools:
        return (
            "Environment: ipython\nTools: brave_search"  # , wolfram_alpha\n\n"
        )
    return "Environment: ipython"

def dates() -> str:
    return f"Cutting Knowledge Date: December 2023\nToday Date: {formatted_date()}\n\n"

def header(text: str, role: str) -> str:
    return f"{text}<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n\n"

def generate_chat_prompt(messages: List[Message]) -> str:
    prompt = f"""{system_header()}{env(False)}
Cutting Knowledge Date: December 2023
Today Date: {formatted_date()}

"""
    for message in messages:
        # Do we care about tool_plan or tool_results or tool_calls here?
        prompt += header(message["content"], role_to_header[message["role"]])

    return prompt