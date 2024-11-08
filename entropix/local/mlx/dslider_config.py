from pydantic import BaseModel
import mlx.core as mx

MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8

class OutlierThreshold(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    bilinear: mx.array
    linear_state_ent: mx.array
    linear_state_std: mx.array
    linear_naked_ent: mx.array
    linear_naked_std: mx.array
    linear_naked_varent: float
    bias: float

    def __post_init__(self):
        for field in ['bilinear', 'linear_state_ent', 'linear_state_std', 'linear_naked_ent', 'linear_naked_std']:
            value = getattr(self, field)
            if not isinstance(value, mx.array):
                object.__setattr__(self, field, mx.array(value))

    def state_dict(self):
        return {k: getattr(self, k) for k in self.__fields__}

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

class ArgmaxThreshold(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    weight: float
    bias: float

    def state_dict(self):
        return {k: getattr(self, k) for k in self.__fields__}

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __hash__(self):
        return hash(
            (
            "ArgmaxThreshold",
            self.weight,
            self.bias,
            )
        )

class DirichletThreshold(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    weight: float
    bias: float

    def state_dict(self):
        return {k: getattr(self, k) for k in self.__fields__}

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __hash__(self):
        return hash(
            (
            "DirichletThreshold",
            self.weight,
            self.bias,
            )
        )

class TargetEntropy(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    linear: mx.array
    linear_inv_temp: mx.array
    bias: float

    def __post_init__(self):
        for field in ['linear', 'linear_inv_temp']:
            value = getattr(self, field)
            if not isinstance(value, mx.array):
                object.__setattr__(self, field, mx.array(value))

    def state_dict(self):
        return {k: getattr(self, k) for k in self.__fields__}

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

class DSConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    #EMWA coefficients
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
    dirichlet_support: mx.array

    # Threshold Detection
    outlier_threshold: OutlierThreshold
    argmax_threshold: ArgmaxThreshold
    dirichlet_threshold: DirichletThreshold
    target_entropy: TargetEntropy

    # Token outlier
    outlier_topk: int

    def __post_init__(self):
        if not isinstance(self.dirichlet_support, mx.array):
            object.__setattr__(self, 'dirichlet_support', mx.array(self.dirichlet_support, dtype = mx.int32))

    def state_dict(self):
        state_dict = {}
        for k in self.__fields__:
            if k in ["outlier_threshold", "argmax_threshold", "dirichlet_threshold", "target_entropy"]:
                state_dict[k] = getattr(self, k).state_dict()
            else:
                state_dict[k] = getattr(self, k)
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict):
        outlier_threshold = OutlierThreshold.from_state_dict(state_dict['outlier_threshold'])
        argmax_threshold = ArgmaxThreshold.from_state_dict(state_dict['argmax_threshold'])
        dirichlet_threshold = DirichletThreshold.from_state_dict(state_dict['dirichlet_threshold'])
        target_entropy = TargetEntropy.from_state_dict(state_dict['target_entropy'])

        return cls(
            **{k: v for k, v in state_dict.items() if k not in ["outlier_threshold", "argmax_threshold", "dirichlet_threshold", "target_entropy"]},
            outlier_threshold = outlier_threshold,
            argmax_threshold = argmax_threshold,
            dirichlet_threshold = dirichlet_threshold,
            target_entropy = target_entropy
        )

    def __hash__(self):
        hashable_items = []
        for k in self.__fields__:
            v = getattr(self, k)
            if isinstance(v, mx.array):
                hashable_items.append((k, tuple(v.shape), str(v.dtype)))
            elif isinstance(v, BaseModel):
                hashable_items.append(hash(v))
            else:
                hashable_items.append(hash((
                    k, v
                )))

        return hash(tuple(hashable_items))

# Default config values
DEFAULT_DS_CONFIG = DSConfig(
    # EMWA coefficients
    emwa_logp_base=1.5,
    emwa_logp_exp_factor=2.5,
    emwa_dir_coeff=0.2,
    emwa_temp_coeff=1,
    emwa_dir_ent_coeff=0.15,
    emwa_ent_scaffold_coeff=0.15,
    emwa_varent_scaffold_coeff=0.15,
    emwa_ent_naked_coeff=0.15,
    emwa_varent_naked_coeff=0.15,
    emwa_topk_ent_naked_coeff=0.15,
    # Token cross entropy coefficients
    token_cross_ent_scaffold_coeff=0.15,
    token_cross_ent_naked_coeff=0.15,
    token_cross_var_scaffold_coeff=0.15,
    token_cross_var_naked_coeff=0.15,
    # Dirichlet parameters
    perturb_base_coeff=0.95,
    perturb_exp_coeff=2.5,
    dirichlet_support=mx.arange(128256),  # this is llama3 vocab size
    # Threshold parameters
    outlier_threshold=OutlierThreshold(
        bilinear=mx.eye(4) * 0.15,  # Increased sensitivity
        linear_state_ent=mx.ones(4) * 0.15,
        linear_state_std=mx.ones(4) * 0.15,
        linear_naked_ent=mx.array(0.15),
        linear_naked_std=mx.array(0.15),
        linear_naked_varent=0.15,
        bias=0.1,  # Added small positive bias
    ),
    argmax_threshold=ArgmaxThreshold(
        weight=1.2,  # Increased from 1.0
        bias=0.1,  # Added small positive bias
    ),
    dirichlet_threshold=DirichletThreshold(
        weight=1.2,  # Increased from 1.0
        bias=0.1,  # Added small positive bias
    ),
    target_entropy=TargetEntropy(
        linear=mx.ones(4) * 0.15,
        linear_inv_temp=mx.ones(1) * 1.2,  # Increased from 1.0
        bias=0.1,  # Added small positive bias
    ),
    # Token outlier parameters
    outlier_topk=5,
)