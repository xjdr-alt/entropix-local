import mlx.core as mx
import pandas as pd
    
def apply_scaling(freqs: mx.array) -> mx.array:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 2048  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: mx.array) -> mx.array:
        wavelen = 2 * mx.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = mx.clip(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = mx.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            mx.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = mx.vmap(scale_freq)(freqs)

    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False, dtype=mx.float32) -> mx.array:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = mx.arange(end, dtype=dtype).reshape(-1,1)  # Shape: (end, 1)
    freqs = freqs.reshape(1,-1)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return mx.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> mx.array:
  mask = None
  if seqlen > 1:
      mask = mx.full((seqlen, seqlen), float("-inf"))
      mask = mx.triu(mask, k=1)
      zeros = mx.zeros((seqlen, start_pos))
      mask = mx.concatenate([zeros, mask], axis=1)
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