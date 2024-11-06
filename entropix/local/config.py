from typing import NamedTuple
from enum import Enum
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerateConfig:
    """Configuration class for text generation parameters.
    
    Attributes:
        prompt (str): The input text to generate from.
        max_tokens (int, optional): Maximum number of tokens to generate.
            Defaults to 600. Range: 1-2048.
        debug (bool, optional): Enable debug output during generation.
            Defaults to True.
        stream (bool, optional): Stream tokens as they're generated.
            Defaults to True.
        csv_file (str, optional): Path to CSV file containing prompts.
            Defaults to None.
    """
    prompt: str = "Tell me a joke"
    max_tokens: Optional[int] = 600
    debug: bool = True
    stream: bool = True
    csv_file: Optional[str] = None

    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.csv_file is None:
            if not isinstance(self.prompt, str):
                raise ValueError("prompt must be a string")
            if not self.prompt.strip():
                raise ValueError("prompt cannot be empty")
            
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int):
                raise ValueError("max_tokens must be an integer")
            if self.max_tokens < 1 or self.max_tokens > 2048:
                raise ValueError("max_tokens must be between 1 and 2048")

    @classmethod
    def help(cls) -> str:
        """Return helpful information about using this configuration class."""
        return """
GenerateConfig Usage:
--------------------
Required:
- prompt (str): The text prompt to generate from
    Example: --config.prompt "Once upon a time"
OR
- csv file (str): path to csv file containing string prompts with column header 'prompts'
    Example: --config.csv_file "prompts.csv"

Optional:
- max_tokens (int): How many tokens to generate (1-2048)
    Default: 600
    Usage: --config.max_tokens 1000
- debug: Toggle debug information during generation
    Default: True
    Usage: --config.debug or --config.no-debug
- stream: Toggle output token streaming
    Default: True
    Usage: --config.stream or --config.no-stream

Example usage:
    python3 -m entropix.local.torch.main --config.prompt "Which number is larger 9.11 or 9.9? be brief in your response" --config.no-stream --config.debug
    or
    python3 -m entropix.local.torch.main --config.csv_file "prompts.csv" --config.stream --config.debug
"""

class EntropixConfig:
    def __init__(self):
        # Sampler state toggles
        ## Low Entropy, Low Varentropy: "flowing with unspoken intent"
        self.state_flowing = False
        ## High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
        self.state_treading = False
        ## Low Entropy, High Varentropy: "exploring forks in the path"
        self.state_exploring = False
        ## High Entropy, High Varentropy: "resampling in the mist"
        self.state_resampling = False

        # Extra sampler state toggles for advanced testing
        self.state_extras_agreement = False
        self.state_extras_interaction_strength = False

@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    max_seq_len: int
    model_size: str

# Define configurations for different models and model sizes
MODEL_CONFIGS = {
    "3B": ModelConfig(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        vocab_size=128256,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        max_seq_len=8192,
        model_size="3B"
    ),
    "1B": ModelConfig(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        max_seq_len=4096,
        model_size="1B"
    )
}

MODEL_IDS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct"

}

MODEL_PATHS = {
    "1B": "entropix/data/1B",
    "3B": "entropix/data/3B"
}

class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool

def get_model_params(config: ModelConfig) -> ModelParams:
    """Create ModelParams from config."""
    return ModelParams(
        n_layers=config.n_layers,
        n_local_heads=config.n_heads,
        n_local_kv_heads=config.n_kv_heads,
        head_dim=config.dim // config.n_heads,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        use_scaled_rope=config.use_scaled_rope
    )

def create_model_params(config: ModelConfig) -> ModelParams:
    """Creates ModelParams from a ModelConfig."""
    return ModelParams(
        n_layers=config.n_layers,
        n_local_heads=config.n_heads,
        n_local_kv_heads=config.n_kv_heads,
        head_dim=config.dim // config.n_heads,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        use_scaled_rope=config.use_scaled_rope,
    )

# Common model parameter configurations
LLAMA_1B_PARAMS = create_model_params(MODEL_CONFIGS["1B"])
LLAMA_3B_PARAMS = create_model_params(MODEL_CONFIGS["3B"])

# Experimental custom config to trigger different sampler states
class SamplerConfig:
    def __init__(self, model_size: str = "1B"):
        """
        Initialize SamplerConfig with specified model size.

        Args:
            model_size: One of "1B", or "3B"
        """
        self.model_size = model_size  # Store model_size as instance variable

        if self.model_size == "1B":
            """
            Configuration for the sampling strategy, including threshold values for various metrics
            and adaptive sampling parameters.
            """
            self.temperature = 0.666
            self.top_p = 0.90
            self.top_k = 27
            self.min_p = 0.03

            self.low_logits_entropy_threshold = 0.01
            self.medium_logits_entropy_threshold = 0.7
            self.high_logits_entropy_threshold = 2.1

            self.low_logits_varentropy_threshold = 0.05
            self.medium_logits_varentropy_threshold = 2.0
            self.high_logits_varentropy_threshold = 5.8

            self.low_attention_entropy_threshold = 11.915
            self.medium_attention_entropy_threshold = 11.921
            self.high_attention_entropy_threshold = 11.926

            self.low_attention_varentropy_threshold = 0.001
            self.medium_attention_varentropy_threshold = 0.0045
            self.high_attention_varentropy_threshold = 0.009

            self.low_agreement_threshold = 1.8e-06
            self.medium_agreement_threshold = 3.8e-06
            self.high_agreement_threshold = 4.8e-06

            self.low_interaction_strength_threshold = 0.18
            self.medium_interaction_strength_threshold = 0.227
            self.high_interaction_strength_threshold = 0.244

            self.high_entropy_attention_offset = 1.3
            self.high_entropy_attention_coefficient = 0.2

            self.low_entropy_interaction_strength_offset = 1.2
            self.low_entropy_interaction_strength_coefficient = 0.3

            self.high_entropy_varentropy_attention_offset = 2.0
            self.high_entropy_varentropy_attention_coefficient = 0.5

            self.n_adaptive_samples = 5

            self.adaptive_temperature_logits_coefficient = 0.3
            self.adaptive_temperature_attention_coefficient = 0.2
            self.adaptive_temperature_agreement_coefficient = 0.2
            self.adaptive_top_p_coefficient = 0.1
            self.adaptive_top_k_interaction_coefficient = 0.3
            self.adaptive_top_k_agreement_coefficient = 0.2
            self.adaptive_min_p_coefficient = 0.5
            self.adaptive_score_logits_entropy_coefficient = 0.1
            self.adaptive_score_attention_entropy_coefficient = 0.2
            self.adaptive_score_logits_varentropy_coefficient = 0.3
            self.adaptive_score_attention_varentropy_coefficient = 0.4
            self.adaptive_score_agreement_coefficient = 0.5
            self.adaptive_score_interaction_strength_coefficient = 0.6


        elif self.model_size == "3B":
            self.temperature = 0.666
            self.top_p = 0.90
            self.top_k = 27
            self.min_p = 0.03

            self.low_logits_entropy_threshold = 0.01
            self.medium_logits_entropy_threshold = 0.7
            self.high_logits_entropy_threshold = 2.1

            self.low_logits_varentropy_threshold = 0.05
            self.medium_logits_varentropy_threshold = 2.0
            self.high_logits_varentropy_threshold = 5.8

            self.low_attention_entropy_threshold = 11.915
            self.medium_attention_entropy_threshold = 11.921
            self.high_attention_entropy_threshold = 11.926

            self.low_attention_varentropy_threshold = 0.001
            self.medium_attention_varentropy_threshold = 0.0045
            self.high_attention_varentropy_threshold = 0.009

            self.low_agreement_threshold = 1.8e-06
            self.medium_agreement_threshold = 3.8e-06
            self.high_agreement_threshold = 4.8e-06

            self.low_interaction_strength_threshold = 0.18
            self.medium_interaction_strength_threshold = 0.227
            self.high_interaction_strength_threshold = 0.244

            self.high_entropy_attention_offset = 1.3
            self.high_entropy_attention_coefficient = 0.2

            self.low_entropy_interaction_strength_offset = 1.2
            self.low_entropy_interaction_strength_coefficient = 0.3

            self.high_entropy_varentropy_attention_offset = 2.0
            self.high_entropy_varentropy_attention_coefficient = 0.5

            self.n_adaptive_samples = 5

            self.adaptive_temperature_logits_coefficient = 0.3
            self.adaptive_temperature_attention_coefficient = 0.2
            self.adaptive_temperature_agreement_coefficient = 0.2
            self.adaptive_top_p_coefficient = 0.1
            self.adaptive_top_k_interaction_coefficient = 0.3
            self.adaptive_top_k_agreement_coefficient = 0.2
            self.adaptive_min_p_coefficient = 0.5
            self.adaptive_score_logits_entropy_coefficient = 0.1
            self.adaptive_score_attention_entropy_coefficient = 0.2
            self.adaptive_score_logits_varentropy_coefficient = 0.3
            self.adaptive_score_attention_varentropy_coefficient = 0.4
            self.adaptive_score_agreement_coefficient = 0.5
            self.adaptive_score_interaction_strength_coefficient = 0.6

        else:
            raise ValueError(f"Invalid model size: {model_size}. Choose from: 1B, 3B")

class SamplerState(Enum):
    FLOWING = "Flowing with unspoken intent"
    TREADING = "Treading carefully, asking clarifying questions"
    EXPLORING = "Exploring forks in the path"
    RESAMPLING = "Resampling in the mist"
    ADAPTIVE = "Adaptive Sampling"