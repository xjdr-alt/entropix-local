from dataclasses import dataclass
import torch
import torch.nn as nn

# Constants
MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8
VOCAB_SIZE = 128256


@dataclass(frozen=True)
class OutlierThreshold:
    bilinear: torch.Tensor  # Shape (4, 4)
    linear_state_ent: torch.Tensor  # Shape (4,)
    linear_state_std: torch.Tensor  # Shape (4,)
    linear_naked_ent: float
    linear_naked_std: float
    linear_naked_varent: float
    bias: float

    def __post_init__(self):
        # Ensure tensors are the correct type
        if not isinstance(self.bilinear, torch.Tensor):
            object.__setattr__(self, "bilinear", torch.tensor(self.bilinear, dtype=torch.float32))
        if not isinstance(self.linear_state_ent, torch.Tensor):
            object.__setattr__(self, "linear_state_ent", torch.tensor(self.linear_state_ent, dtype=torch.float32))
        if not isinstance(self.linear_state_std, torch.Tensor):
            object.__setattr__(self, "linear_state_std", torch.tensor(self.linear_state_std, dtype=torch.float32))

    def to(self, device):
        return OutlierThreshold(
            bilinear=self.bilinear.to(device),
            linear_state_ent=self.linear_state_ent.to(device),
            linear_state_std=self.linear_state_std.to(device),
            linear_naked_ent=self.linear_naked_ent,
            linear_naked_std=self.linear_naked_std,
            linear_naked_varent=self.linear_naked_varent,
            bias=self.bias,
        )

    def state_dict(self):
        return {
            "bilinear": self.bilinear,
            "linear_state_ent": self.linear_state_ent,
            "linear_state_std": self.linear_state_std,
            "linear_naked_ent": self.linear_naked_ent,
            "linear_naked_std": self.linear_naked_std,
            "linear_naked_varent": self.linear_naked_varent,
            "bias": self.bias,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __hash__(self):
        return hash(
            (
                "OutlierThreshold",
                tuple(self.bilinear.shape),
                str(self.bilinear.dtype),
                tuple(self.linear_state_ent.shape),
                str(self.linear_state_ent.dtype),
                tuple(self.linear_state_std.shape),
                str(self.linear_state_std.dtype),
                self.linear_naked_ent,
                self.linear_naked_std,
                self.linear_naked_varent,
                self.bias,
            )
        )


@dataclass(frozen=True)
class ArgmaxThreshold:
    weight: float
    bias: float

    def to(self, device):
        return self  # No tensors to move

    def state_dict(self):
        return {"weight": self.weight, "bias": self.bias}

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __hash__(self):
        return hash((self.weight, self.bias))


@dataclass(frozen=True)
class DirichletThreshold:
    weight: float
    bias: float

    def to(self, device):
        return self  # No tensors to move

    def state_dict(self):
        return {"weight": self.weight, "bias": self.bias}

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __hash__(self):
        return hash((self.weight, self.bias))


@dataclass(frozen=True)
class TargetEntropy:
    linear: torch.Tensor  # Shape (4,)
    linear_inv_temp: torch.Tensor  # Shape (batch_size,)
    bias: float

    def __post_init__(self):
        # Ensure tensors are the correct type
        if not isinstance(self.linear, torch.Tensor):
            object.__setattr__(self, "linear", torch.tensor(self.linear, dtype=torch.float32))
        if not isinstance(self.linear_inv_temp, torch.Tensor):
            object.__setattr__(self, "linear_inv_temp", torch.tensor(self.linear_inv_temp, dtype=torch.float32))

    def to(self, device):
        return TargetEntropy(
            linear=self.linear.to(device),
            linear_inv_temp=self.linear_inv_temp.to(device),
            bias=self.bias,
        )

    def state_dict(self):
        return {"linear": self.linear, "linear_inv_temp": self.linear_inv_temp, "bias": self.bias}

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __hash__(self):
        return hash(
            (
                "TargetEntropy",
                tuple(self.linear.shape),
                str(self.linear.dtype),
                tuple(self.linear_inv_temp.shape),
                str(self.linear_inv_temp.dtype),
                self.bias,
            )
        )


@dataclass(frozen=True, eq=True)
class DSConfig:
    # EMWA coefficients
    emwa_logp_base: float
    emwa_logp_exp_factor: float
    emwa_dir_coeff: float
    emwa_temp_coeff: float
    emwa_dir_ent_coeff: float
    emwa_ent_scaffold_coeff: float
    emwa_varent_scaffold_coeff: float
    emwa_ent_naked_coeff: float
    emwa_varent_naked_coeff: float
    emwa_topk_ent_naked_coeff: float

    # Token cross entropy coefficients
    token_cross_ent_scaffold_coeff: float
    token_cross_ent_naked_coeff: float
    token_cross_var_scaffold_coeff: float
    token_cross_var_naked_coeff: float

    # Dirichlet parameters
    perturb_base_coeff: float
    perturb_exp_coeff: float
    """
    dirichlet_support is a subset of the vocabulary of your model.
    recommended tuning:
    1. sample autoregressively conditioned on random hidden state prompts
    2. take the empirical average of logprobs across position and prompts
    3. the support is all logprobs lying above the noise threshold (see normalize_logits in dslider.py)
    """
    dirichlet_support: torch.Tensor

    # noise floor for logits normalization
    noise_floor: float

    # Threshold parameters
    outlier_threshold: OutlierThreshold
    argmax_threshold: ArgmaxThreshold
    dirichlet_threshold: DirichletThreshold
    target_entropy: TargetEntropy

    # Token outlier
    outlier_topk: int

    def __post_init__(self):
        # Ensure tensor is the correct type
        if not isinstance(self.dirichlet_support, torch.Tensor):
            object.__setattr__(self, "dirichlet_support", torch.tensor(self.dirichlet_support, dtype=torch.int64))

    def to(self, device):
        return DSConfig(
            # EMWA coefficients
            emwa_logp_base=self.emwa_logp_base,
            emwa_logp_exp_factor=self.emwa_logp_exp_factor,
            emwa_dir_coeff=self.emwa_dir_coeff,
            emwa_temp_coeff=self.emwa_temp_coeff,
            emwa_dir_ent_coeff=self.emwa_dir_ent_coeff,
            emwa_ent_scaffold_coeff=self.emwa_ent_scaffold_coeff,
            emwa_varent_scaffold_coeff=self.emwa_varent_scaffold_coeff,
            emwa_ent_naked_coeff=self.emwa_ent_naked_coeff,
            emwa_varent_naked_coeff=self.emwa_varent_naked_coeff,
            emwa_topk_ent_naked_coeff=self.emwa_topk_ent_naked_coeff,
            # Token cross entropy coefficients
            token_cross_ent_scaffold_coeff=self.token_cross_ent_scaffold_coeff,
            token_cross_ent_naked_coeff=self.token_cross_ent_naked_coeff,
            token_cross_var_scaffold_coeff=self.token_cross_var_scaffold_coeff,
            token_cross_var_naked_coeff=self.token_cross_var_naked_coeff,
            # Dirichlet parameters
            perturb_base_coeff=self.perturb_base_coeff,
            perturb_exp_coeff=self.perturb_exp_coeff,
            dirichlet_support=self.dirichlet_support.to(device),
            noise_floor=self.noise_floor,
            # Threshold parameters
            outlier_threshold=self.outlier_threshold.to(device),
            argmax_threshold=self.argmax_threshold.to(device),
            dirichlet_threshold=self.dirichlet_threshold.to(device),
            target_entropy=self.target_entropy.to(device),
            # Token outlier
            outlier_topk=self.outlier_topk,
        )

    def state_dict(self):
        """Convert the config to a state dict for saving/loading"""
        return {
            # EMWA coefficients
            "emwa_logp_base": self.emwa_logp_base,
            "emwa_logp_exp_factor": self.emwa_logp_exp_factor,
            "emwa_dir_coeff": self.emwa_dir_coeff,
            "emwa_temp_coeff": self.emwa_temp_coeff,
            "emwa_dir_ent_coeff": self.emwa_dir_ent_coeff,
            "emwa_ent_scaffold_coeff": self.emwa_ent_scaffold_coeff,
            "emwa_varent_scaffold_coeff": self.emwa_varent_scaffold_coeff,
            "emwa_ent_naked_coeff": self.emwa_ent_naked_coeff,
            "emwa_varent_naked_coeff": self.emwa_varent_naked_coeff,
            "emwa_topk_ent_naked_coeff": self.emwa_topk_ent_naked_coeff,
            # Token cross entropy coefficients
            "token_cross_ent_scaffold_coeff": self.token_cross_ent_scaffold_coeff,
            "token_cross_ent_naked_coeff": self.token_cross_ent_naked_coeff,
            "token_cross_var_scaffold_coeff": self.token_cross_var_scaffold_coeff,
            "token_cross_var_naked_coeff": self.token_cross_var_naked_coeff,
            # Dirichlet parameters
            "perturb_base_coeff": self.perturb_base_coeff,
            "perturb_exp_coeff": self.perturb_exp_coeff,
            "dirichlet_support": self.dirichlet_support,
            # Threshold parameters
            "outlier_threshold": self.outlier_threshold.state_dict(),
            "argmax_threshold": self.argmax_threshold.state_dict(),
            "dirichlet_threshold": self.dirichlet_threshold.state_dict(),
            "target_entropy": self.target_entropy.state_dict(),
            # Token outlier
            "outlier_topk": self.outlier_topk,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Create a new instance from a state dict"""
        # First reconstruct the nested objects
        outlier_threshold = OutlierThreshold.from_state_dict(state_dict.pop("outlier_threshold"))
        argmax_threshold = ArgmaxThreshold.from_state_dict(state_dict.pop("argmax_threshold"))
        dirichlet_threshold = DirichletThreshold.from_state_dict(state_dict.pop("dirichlet_threshold"))
        target_entropy = TargetEntropy.from_state_dict(state_dict.pop("target_entropy"))

        # Create the config with all parameters
        return cls(
            outlier_threshold=outlier_threshold,
            argmax_threshold=argmax_threshold,
            dirichlet_threshold=dirichlet_threshold,
            target_entropy=target_entropy,
            **state_dict,
        )

    def __hash__(self):
        """Static hash implementation that avoids hashing tensor values"""
        hashable_items = []
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                # Only hash shape and dtype for tensors
                hashable_items.append(hash((str(field.name), tuple(value.shape), str(value.dtype))))
            elif isinstance(value, (OutlierThreshold, ArgmaxThreshold, DirichletThreshold, TargetEntropy)):
                # Use the class's hash method
                hashable_items.append(hash(value))
            else:
                # For primitive types
                hashable_items.append(hash((str(field.name), value)))
        return hash(tuple(hashable_items))


# Default config values
DEFAULT_DS_CONFIG = DSConfig(
    emwa_logp_base=4.0,
    emwa_logp_exp_factor=3.0,
    emwa_dir_coeff=0.70,
    emwa_temp_coeff=0.70,
    emwa_dir_ent_coeff=0.70,
    emwa_ent_scaffold_coeff=0.70,
    emwa_varent_scaffold_coeff=0.70,
    emwa_ent_naked_coeff=0.70,
    emwa_varent_naked_coeff=0.70,
    emwa_topk_ent_naked_coeff=0.70,
    token_cross_ent_scaffold_coeff=0.65,
    token_cross_ent_naked_coeff=0.65,
    token_cross_var_scaffold_coeff=0.75,
    token_cross_var_naked_coeff=0.65,
    perturb_base_coeff=10.0,
    perturb_exp_coeff=1.0,
    dirichlet_support=torch.arange(VOCAB_SIZE),
    noise_floor=-12.0,
    # Threshold parameters
    outlier_threshold=OutlierThreshold(
        bilinear=torch.ones((4, 4)) * 1.3,  # Increased sensitivity
        linear_state_ent=torch.ones(4) * 0.8,
        linear_state_std=torch.ones(4) * 0.8,
        linear_naked_ent=1.2,
        linear_naked_std=1.2,
        linear_naked_varent=1.2,
        bias=0.0,
    ),
    argmax_threshold=ArgmaxThreshold(
        weight=1.0,
        bias=5.0,
    ),
    dirichlet_threshold=DirichletThreshold(weight=1.0, bias=5.0),
    target_entropy=TargetEntropy(
        linear=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        linear_inv_temp=torch.ones(1) * 8.0,
        bias=0.0,
    ),
    outlier_topk=3,
)
