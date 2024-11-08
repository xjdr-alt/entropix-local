import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange
from scipy.stats import dirichlet
import scipy.special as sp

from typing import NamedTuple, Tuple
from entropix.local.mlx.dslider_config import EPS, MAX_TEMP, MIN_TEMP, DSConfig
from entropix.local.mlx.dslider_utils import fit_dirichlet, temp_tune

def lgamma(alpha: mx.array):
    lgamma = sp.loggamma(alpha)
    return mx.array(lgamma)

def digamma(alpha: mx.array):
    digamma = sp.digamma(alpha)
    return mx.array(digamma)

def polygamma(n: int, alpha: mx.array):
    polygamma = sp.polygamma(n, alpha)
    return mx.array(polygamma)

def kl_divergence(logp: mx.array, logq: mx.array) -> mx.array:
    """
    Computes the KL divergence between two sets of log probabilities.
    """
    p = mx.exp(logp)
    kl_elements = p * (logp - logq)
    kl_elements = mx.where(p > 0, kl_elements, 0)
    return mx.sum(kl_elements, axis = -1)

def ent_varent(logp: mx.array):
    """
    Compute entropy and variance of entropy from log-probabilities
    """
    p = mx.exp(logp)
    ent = -mx.sum(p * logp, axis = -1)
    diff = logp + rearrange(ent, "v -> v 1")
    varent = mx.sum(p * diff **2, axis = -1)
    return ent, varent

def dirichlet_expectation(alpha: mx.array) -> mx.array:
    """
    Computes the Dirichlet expectation.
    """
    return alpha / mx.sum(alpha, axis = -1, keepdims = True)

def sample_dirichlet(alpha: mx.array, key: mx.array) -> mx.array:
    """
    Samples from a Dirichlet distribution.
    """
    return mx.array(dirichlet(alpha, int(key[0])))

class DSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""

    emwa_dir: mx.array
    emwa_logp_dir_supp: mx.array
    emwa_temp: mx.array
    emwa_ent_scaffold: mx.array
    emwa_ent_naked: mx.array
    emwa_varent_scaffold: mx.array
    emwa_varent_naked: mx.array
    token_cross_ent_scaffold: mx.array
    token_cross_ent_naked: mx.array
    token_cross_var_scaffold: mx.array
    token_cross_var_naked: mx.array
    emwa_dir_ent: mx.array
    emwa_topk_ent_naked: mx.array

def dirichlet_expected_entropy(alpha: mx.array) -> mx.array:
    """
    Compute the expected entropy of a Dirichlet distribution.
    """
    alpha_sum = mx.sum(alpha, axis = -1, keepdims = True)
    K = alpha.shape[-1]

    # ln B(alpha) term
    log_beta = mx.sum(lgamma(alpha), axis = -1) - lgamma(rearrange(alpha_sum, "v -> v 1"))

    # (alpha_0 - K) * ψ(alpha_0) term
    digamma_sum = digamma(alpha_sum)
    second_term = (rearrange(alpha_sum, "1 v -> v") - K) * rearrange(digamma_sum, "1 v -> v")

    # -sum((alpha_j - 1) * ψ(alpha_j)) term
    digamma_alpha = digamma(alpha)
    third_term = -mx.sum((alpha - 1) * digamma_alpha, axis = -1)

    return log_beta + second_term + third_term

def dirichlet_log_likelihood_from_logprob(logprobs: mx.array, alpha: mx.array) -> mx.array:
    """
    Compute the log likelihood of a set of log probabilities under a Dirichlet distribution.
    """
    return (
        mx.sum((alpha - 1) * logprobs, axis = -1)
        - lgamma(mx.sum(alpha, axis = -1))
        + mx.sum(lgamma(alpha), axis = -1)
    )

def dirichlet_expected_varentropy(alpha: mx.array) -> mx.array:
    """
    Compute the expected variance of entropy of a Dirichlet distribution.
    """
    alpha_sum = mx.sum(alpha, axis = -1, keepdims = True)

    # E[Xᵢ] = αᵢ / α₀
    expected_x = alpha / alpha_sum

    # E[X] = α / α_0
    digamma_alpha = digamma(alpha)
    trigamma_alpha = polygamma(1, alpha)

    squared_plus_deriv = digamma_alpha ** 2 + trigamma_alpha

    return mx.sum(expected_x * squared_plus_deriv, axis = -1)

def initialize_state(bsz: int, vsz: int, config: DSConfig) -> DSState:
    """Initialize the DSState with specified dtype."""
    state = DSState(
        emwa_dir=mx.ones((bsz, config.dirichlet_support.size)),
        emwa_logp_dir_supp=mx.zeros((bsz, config.dirichlet_support.size)),
        emwa_temp=mx.ones((bsz,)),
        emwa_ent_scaffold=mx.zeros((bsz,)),
        emwa_ent_naked=mx.zeros((bsz,)),
        emwa_varent_scaffold=mx.zeros((bsz,)),
        emwa_varent_naked=mx.zeros((bsz,)),
        token_cross_ent_scaffold=mx.zeros((bsz,)),
        token_cross_ent_naked=mx.zeros((bsz,)),
        token_cross_var_scaffold=mx.zeros((bsz,)),
        token_cross_var_naked=mx.zeros((bsz,)),
        emwa_dir_ent=mx.zeros((bsz,)),
        emwa_topk_ent_naked=mx.zeros((bsz,)),
    )
    return state

def categorical_sample(probs: mx.array, key: mx.array) -> mx.array:
    """
    Sample from a categorical distribution using the Gumbel-Max trick.
    """
    uniform = mx.random.uniform(0, 1, shape=probs.shape, key=key)
    gumbel = -mx.log(-mx.log(uniform + 1e-10) + 1e-10)
    return mx.argmax(mx.log(probs + 1e-10) + gumbel, axis=-1)

def adaptive_dirichlet_step(
    state: DSState,
    logits: mx.array,
    config: DSConfig,
    wild: bool = True,
) -> Tuple[DSState, mx.array]:
    """Single step of the Adaptive Dirichlet Sampler."""
    dtype = logits.dtype
    bsz, _ = logits.shape
    output_tokens = mx.zeros((bsz,), dtype=mx.int32)
    EPS_tensor = mx.array(EPS, dtype=dtype)
    # normalize logits
    naked_log_probs = normalize_logits(logits)
    # update naked entropy rate
    naked_ent, naked_varent = ent_varent(naked_log_probs)
    new_emwa_ent_naked = (
        config.emwa_ent_naked_coeff * naked_ent + (1 - config.emwa_ent_naked_coeff) * state.emwa_ent_naked
    )
    new_emwa_varent_naked = (
        config.emwa_varent_naked_coeff * naked_varent + (1 - config.emwa_varent_naked_coeff) * state.emwa_varent_naked
    )
    # entropy and varentropy vectors - shape (bsz, 4)
    state_ent = mx.stack(
        [
            state.token_cross_ent_scaffold,
            state.token_cross_ent_naked,
            state.emwa_ent_scaffold,
            state.emwa_ent_naked,
        ],
        axis=1,
    )
    state_std = mx.sqrt(
        mx.stack(
            [
                state.token_cross_var_scaffold,
                state.token_cross_var_naked,
                state.emwa_varent_scaffold,
                state.emwa_varent_naked,
            ],
            axis=1,
        )
    )
    outlier_threshold = compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config)
    outlier_mask = outlier_threshold > 0
    # extract topk
    k = min(config.outlier_topk, naked_log_probs.shape[-1])

    sorted_indices = mx.argsort(-naked_log_probs, axis=-1)
    probs_sort = mx.take_along_axis(-naked_log_probs, sorted_indices, axis=-1)
    probs_sort = -probs_sort
    topk_logits = probs_sort[..., :k]
    topk_indices = sorted_indices[..., :k]
    # update emwa topk entropy
    topk_logprobs = normalize_logits(topk_logits)
    naked_topk_ent, _ = ent_varent(topk_logprobs)
    new_emwa_topk_ent_naked = (
        config.emwa_topk_ent_naked_coeff * naked_topk_ent
        + (1 - config.emwa_topk_ent_naked_coeff) * state.emwa_topk_ent_naked
    )
    # argmax policy for concentrated inliers
    argmax_threshold = config.argmax_threshold.weight * state.emwa_topk_ent_naked + config.argmax_threshold.bias
    argmax_mask = (~outlier_mask) & (naked_topk_ent < argmax_threshold)
    # Get indices of maximum probabilities within top-k
    argmax_indices = mx.argmax(topk_logprobs, axis=-1)
    # Map these indices back to the original token space using topk_indices

    batch_indices = mx.arange(bsz)
    argmax_tokens = topk_indices[batch_indices, argmax_indices]
    # Only use these tokens where argmax_mask is True
    output_tokens = mx.where(argmax_mask, argmax_tokens, output_tokens)
    # topk temperature tuning policy for dispersed inliers
    inlier_sampling_indices = (~outlier_mask) & (~argmax_mask)
    
    # Now use temp_clipped for inlier sampling
    inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked, mx.ones_like(state.emwa_temp))
    temp_clipped = mx.clip(inlier_sampling_temp, MIN_TEMP, MAX_TEMP)
    sampling_inlier_probs = mx.softmax(topk_logprobs / mx.expand_dims(temp_clipped, -1), axis=-1)
    key = mx.random.key(1337)  
    sampling_inlier_choices = categorical_sample(sampling_inlier_probs, key)

    sampling_inlier_tokens = topk_indices[batch_indices, sampling_inlier_choices]
    output_tokens = mx.where(inlier_sampling_indices, sampling_inlier_tokens, output_tokens)
    # target entropy = affine function of state_ent and inverse emwa temperature
    target_entropy = (
        mx.matmul(state_ent, config.target_entropy.linear.reshape(-1, 1)).squeeze(-1)
        + mx.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, axis=-1)
        + config.target_entropy.bias
    )
    temp, _, _ = temp_tune(naked_log_probs.astype(mx.float32), target_entropy, state.emwa_temp)
    # update emwa temperature
    new_emwa_temp = config.emwa_temp_coeff * temp + (1 - config.emwa_temp_coeff) * state.emwa_temp
    # tune temperature and update emwa logp on dirichlet support
    temp_clipped = mx.clip(temp, MIN_TEMP, MAX_TEMP)
    tuned_logprobs = normalize_logits(naked_log_probs / mx.expand_dims(temp_clipped, -1))
    # update emwa logp and dirichlet parameters
    dir_support_logp = normalize_logits(tuned_logprobs[:, config.dirichlet_support])
    new_emwa_dir, new_emwa_logp_dir_sup, kl = update_dirichlet_params(dir_support_logp, state, config)
    # update Dirichlet entropy
    dir_log_likelihood = dirichlet_log_likelihood_from_logprob(dir_support_logp, state.emwa_dir)
    new_emwa_dir_ent = (
        config.emwa_dir_ent_coeff * (-dir_log_likelihood) + (1 - config.emwa_dir_ent_coeff) * state.emwa_dir_ent
    )
    dirichlet_threshold = config.dirichlet_threshold.weight * state.emwa_dir_ent + config.dirichlet_threshold.bias
    use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
    # below dirichlet threshold, interpolate and sample
    # compute perturbation coefficient
    dir_expectation = dirichlet_expectation(state.emwa_dir)
    kl_div = dirichlet_expected_entropy(state.emwa_dir) - mx.sum(dir_expectation * dir_support_logp, axis=-1)
    perturb_coeff = 1 - mx.power(config.perturb_base_coeff, -config.perturb_exp_coeff / (kl_div + EPS_tensor))
    # Calculate interpolated probabilities for the support tokens
    interpolated_probs = perturb_coeff.expand_dims(-1) * dir_expectation + (1 - perturb_coeff.expand_dims(-1)) * mx.exp(
        dir_support_logp
    )
    # For use_dirichlet case: sample from support space then map back
    interpolated_choices = mx.argmax(interpolated_probs, axis=-1)
    dirichlet_tokens = config.dirichlet_support[interpolated_choices]
    output_tokens = mx.where(use_dirichlet, dirichlet_tokens, output_tokens)
    # above dirichlet threshold you're ngmi
    if wild:
        # sample from random dirichlet distributed
        sampled_probs = sample_dirichlet(new_emwa_dir.float())
        ood_choices = mx.argmax(sampled_probs, axis=-1)
        ood_tokens = config.dirichlet_support[ood_choices]
    else:
        # sample from the pure tuned logprobs
        support_probs = mx.softmax(tuned_logprobs, axis=-1)
        support_choices = categorical_sample(support_probs, key)
        ood_tokens = config.dirichlet_support[support_choices]
    # Update output tokens where appropriate
    output_tokens = mx.where(outlier_mask & ~use_dirichlet, ood_tokens, output_tokens)
    # update scaffold entropy rate
    scaffold_ent, scaffold_varent = ent_varent(mx.log(interpolated_probs + EPS_tensor))
    new_emwa_ent_scaffold = (
        config.emwa_ent_scaffold_coeff * scaffold_ent + (1 - config.emwa_ent_scaffold_coeff) * state.emwa_ent_scaffold
    )
    new_emwa_varent_scaffold = (
        config.emwa_varent_scaffold_coeff * scaffold_varent
        + (1 - config.emwa_varent_scaffold_coeff) * state.emwa_varent_scaffold
    )
    # update token cross entropies
    batch_indices = mx.arange(bsz)

    scaffold_token_logprob = mx.log(interpolated_probs[batch_indices, output_tokens] + EPS_tensor)
    naked_token_logprob = mx.log(naked_log_probs[batch_indices, output_tokens] + EPS_tensor)
    (
        new_token_cross_ent_scaffold,
        new_token_cross_ent_naked,
        new_token_cross_var_scaffold,
        new_token_cross_var_naked,
    ) = update_token_cross_entropies(state, scaffold_token_logprob, naked_token_logprob, config)
    # assemble new state
    new_state = DSState(
        emwa_dir=new_emwa_dir,
        emwa_logp_dir_supp=new_emwa_logp_dir_sup,
        emwa_temp=new_emwa_temp,
        emwa_ent_scaffold=new_emwa_ent_scaffold,
        emwa_ent_naked=new_emwa_ent_naked,
        emwa_varent_scaffold=new_emwa_varent_scaffold,
        emwa_varent_naked=new_emwa_varent_naked,
        token_cross_ent_scaffold=new_token_cross_ent_scaffold,
        token_cross_ent_naked=new_token_cross_ent_naked,
        token_cross_var_scaffold=new_token_cross_var_scaffold,
        token_cross_var_naked=new_token_cross_var_naked,
        emwa_dir_ent=new_emwa_dir_ent,
        emwa_topk_ent_naked=new_emwa_topk_ent_naked,
    )
    return new_state, output_tokens, kl


def normalize_logits(logits: mx.array) -> mx.array:
    """Normalize logits to log probabilities with numerical stability."""
    shifted = logits - mx.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    return log_probs


def update_token_cross_entropies(
    state: DSState,
    scaffold_token_logprob: mx.array,
    naked_token_logprob: mx.array,
    config: DSConfig,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Update token cross entropy statistics."""
    token_cross_ent_scaffold = (
        config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob)
        + (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
    )
    token_cross_ent_naked = (
        config.token_cross_ent_naked_coeff * (-naked_token_logprob)
        + (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
    )
    token_cross_var_scaffold = (
        config.token_cross_var_scaffold_coeff * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2
        + (1 - config.token_cross_var_scaffold_coeff) * state.token_cross_var_scaffold
    )
    token_cross_var_naked = (
        config.token_cross_var_naked_coeff * (token_cross_ent_naked - naked_token_logprob) ** 2
        + (1 - config.token_cross_var_naked_coeff) * state.token_cross_var_naked
    )
    return (
        token_cross_ent_scaffold,
        token_cross_ent_naked,
        token_cross_var_scaffold,
        token_cross_var_naked,
    )


def compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config):
    return (
        mx.einsum("bi,ij,bj->b", state_ent, config.outlier_threshold.bilinear, state_std)
        + mx.einsum("bi,i->b", state_ent, config.outlier_threshold.linear_state_ent)
        + mx.einsum("bi,i->b", state_std, config.outlier_threshold.linear_state_std)
        + naked_ent * config.outlier_threshold.linear_naked_ent
        + naked_varent * config.outlier_threshold.linear_naked_varent
        + config.outlier_threshold.bias
    )


def update_dirichlet_params(dir_support_logp, state, config):
    kl = kl_divergence(dir_support_logp, state.emwa_logp_dir_supp)
    emwa_logp_coeff = mx.expand_dims(config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS)), -1)
    new_emwa_logp_dir_sup = emwa_logp_coeff * dir_support_logp + (1 - emwa_logp_coeff) * state.emwa_logp_dir_supp
    new_dir_params, _, _ = fit_dirichlet(new_emwa_logp_dir_sup, state.emwa_dir)
    return new_dir_params, new_emwa_logp_dir_sup, kl
