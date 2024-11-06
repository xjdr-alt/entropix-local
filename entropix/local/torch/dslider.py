import torch
import torch.nn.functional as F
from typing import NamedTuple, Tuple
from .dslider_config import EPS, MAX_TEMP, MIN_TEMP, DSConfig
from .dslider_utils import fit_dirichlet, temp_tune


def kl_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two log probability distributions."""
    p = torch.exp(logp)
    kl_elements = p * (logp - logq)
    kl_elements = torch.where(p > 0, kl_elements, torch.zeros_like(p))
    kl = torch.sum(kl_elements, dim=-1)
    return kl


def ent_varent(logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute entropy and variance from log probabilities."""
    p = torch.exp(logp)
    ent = -torch.sum(p * logp, dim=-1)
    diff = logp + ent.unsqueeze(-1)
    varent = torch.sum(p * diff**2, dim=-1)
    return ent, varent


def dirichlet_expectation(alpha: torch.Tensor) -> torch.Tensor:
    """Compute the expectation E[X|X~Dir(alpha)]"""
    alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)
    return alpha / alpha_sum


def sample_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """Sample from a Dirichlet distribution."""
    dirichlet_dist = torch.distributions.Dirichlet(alpha)
    return dirichlet_dist.sample()


class DSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""

    emwa_dir: torch.Tensor
    emwa_logp_dir_supp: torch.Tensor
    emwa_temp: torch.Tensor
    emwa_ent_scaffold: torch.Tensor
    emwa_ent_naked: torch.Tensor
    emwa_varent_scaffold: torch.Tensor
    emwa_varent_naked: torch.Tensor
    token_cross_ent_scaffold: torch.Tensor
    token_cross_ent_naked: torch.Tensor
    token_cross_var_scaffold: torch.Tensor
    token_cross_var_naked: torch.Tensor
    emwa_dir_ent: torch.Tensor
    emwa_topk_ent_naked: torch.Tensor
    


def dirichlet_expected_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """Compute the expected entropy of a Dirichlet distribution."""
    alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)  # alpha_0
    K = alpha.shape[-1]

    # ln B(alpha) term
    log_beta = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha_sum.squeeze(-1))

    # (alpha_0 - K) * ψ(alpha_0) term
    digamma_sum = torch.digamma(alpha_sum)
    second_term = (alpha_sum.squeeze(-1) - K) * digamma_sum.squeeze(-1)

    # -sum((alpha_j - 1) * ψ(alpha_j)) term
    digamma_alpha = torch.digamma(alpha)
    third_term = -torch.sum((alpha - 1) * digamma_alpha, dim=-1)

    return log_beta + second_term + third_term


def dirichlet_log_likelihood_from_logprob(logprobs: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute log probability of probs under Dirichlet(alpha)."""
    return (
        torch.sum((alpha - 1.0) * logprobs, dim=-1)
        - torch.lgamma(torch.sum(alpha, dim=-1))
        + torch.sum(torch.lgamma(alpha), dim=-1)
    )


def dirichlet_expected_varentropy(alpha: torch.Tensor) -> torch.Tensor:
    """Compute the expected varentropy E[∑ᵢ ln(Xᵢ)² * Xᵢ] of a Dirichlet distribution."""
    alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)  # α₀

    # E[Xᵢ] = αᵢ / α₀
    expected_x = alpha / alpha_sum

    # ψ(αᵢ)² + ψ₁(αᵢ) term
    digamma_alpha = torch.digamma(alpha)
    trigamma_alpha = torch.polygamma(1, alpha)
    squared_plus_deriv = digamma_alpha**2 + trigamma_alpha

    # Sum over dimensions: ∑ᵢ (αᵢ/α₀) * (ψ₁(αᵢ) + ψ(αᵢ)²)
    return torch.sum(expected_x * squared_plus_deriv, dim=-1)


def initialize_state(bsz: int, vsz: int, config: DSConfig, device, dtype=torch.float32) -> DSState:
    """Initialize the DSState with specified dtype."""
    state = DSState(
        emwa_dir=torch.ones((bsz, config.dirichlet_support.numel()), dtype=dtype, device=device),
        emwa_logp_dir_supp=torch.zeros((bsz, config.dirichlet_support.numel()), dtype=dtype, device=device),
        emwa_temp=torch.ones((bsz,), dtype=dtype, device=device),
        emwa_ent_scaffold=torch.zeros((bsz,), dtype=dtype, device=device),
        emwa_ent_naked=torch.zeros((bsz,), dtype=dtype, device=device),
        emwa_varent_scaffold=torch.zeros((bsz,), dtype=dtype, device=device),
        emwa_varent_naked=torch.zeros((bsz,), dtype=dtype, device=device),
        token_cross_ent_scaffold=torch.zeros((bsz,), dtype=dtype, device=device),
        token_cross_ent_naked=torch.zeros((bsz,), dtype=dtype, device=device),
        token_cross_var_scaffold=torch.zeros((bsz,), dtype=dtype, device=device),
        token_cross_var_naked=torch.zeros((bsz,), dtype=dtype, device=device),
        emwa_dir_ent=torch.zeros((bsz,), dtype=dtype, device=device),
        emwa_topk_ent_naked=torch.zeros((bsz,), dtype=dtype, device=device),
    )
    return state


def adaptive_dirichlet_step(
    state: DSState,
    logits: torch.Tensor,
    config: DSConfig,
    wild: bool = True,
) -> Tuple[DSState, torch.Tensor]:
    """Single step of the Adaptive Dirichlet Sampler."""
    dtype = logits.dtype
    device = logits.device
    bsz, _ = logits.shape
    output_tokens = torch.zeros(bsz, dtype=torch.long, device=device)
    EPS_tensor = torch.tensor(EPS, dtype=dtype, device=device)
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
    state_ent = torch.stack(
        [
            state.token_cross_ent_scaffold,
            state.token_cross_ent_naked,
            state.emwa_ent_scaffold,
            state.emwa_ent_naked,
        ],
        dim=1,
    )
    state_std = torch.sqrt(
        torch.stack(
            [
                state.token_cross_var_scaffold,
                state.token_cross_var_naked,
                state.emwa_varent_scaffold,
                state.emwa_varent_naked,
            ],
            dim=1,
        )
    )
    outlier_threshold = compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config)
    outlier_mask = outlier_threshold > 0
    # extract topk
    topk_logits, topk_indices = torch.topk(naked_log_probs, config.outlier_topk, dim=-1)
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
    argmax_indices = torch.argmax(topk_logprobs, dim=-1)
    # Map these indices back to the original token space using topk_indices
    argmax_tokens = torch.gather(topk_indices, 1, argmax_indices.unsqueeze(1)).squeeze(1)
    # Only use these tokens where argmax_mask is True
    output_tokens = torch.where(argmax_mask, argmax_tokens, output_tokens)
    # topk temperature tuning policy for dispersed inliers
    inlier_sampling_indices = (~outlier_mask) & (~argmax_mask)
    # Handle less confident inliers by sampling with entropy-tuned temperature
    inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked)
    temp_clipped = torch.clamp(inlier_sampling_temp, MIN_TEMP, MAX_TEMP)
    sampling_inlier_probs = torch.softmax(topk_logprobs / temp_clipped.unsqueeze(-1), dim=-1)
    sampling_inlier_choices = torch.multinomial(sampling_inlier_probs, num_samples=1).squeeze(1)
    sampling_inlier_tokens = torch.gather(topk_indices, 1, sampling_inlier_choices.unsqueeze(1)).squeeze(1)
    output_tokens = torch.where(inlier_sampling_indices, sampling_inlier_tokens, output_tokens)
    # target entropy = affine function of state_ent and inverse emwa temperature
    target_entropy = (
        torch.matmul(state_ent, config.target_entropy.linear)
        + torch.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, dim=-1)
        + config.target_entropy.bias
    )
    temp, _, _ = temp_tune(naked_log_probs.float(), target_entropy)
    # update emwa temperature
    new_emwa_temp = config.emwa_temp_coeff * temp + (1 - config.emwa_temp_coeff) * state.emwa_temp
    # tune temperature and update emwa logp on dirichlet support
    temp_clipped = torch.clamp(temp, MIN_TEMP, MAX_TEMP)
    tuned_logprobs = normalize_logits(naked_log_probs / temp_clipped.unsqueeze(-1))
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
    kl_div = dirichlet_expected_entropy(state.emwa_dir) - torch.sum(dir_expectation * dir_support_logp, dim=-1)
    perturb_coeff = 1 - torch.pow(config.perturb_base_coeff, -config.perturb_exp_coeff / (kl_div + EPS_tensor))
    # Calculate interpolated probabilities for the support tokens
    interpolated_probs = perturb_coeff.unsqueeze(-1) * dir_expectation + (1 - perturb_coeff.unsqueeze(-1)) * torch.exp(
        dir_support_logp
    )
    # For use_dirichlet case: sample from support space then map back
    interpolated_choices = torch.argmax(interpolated_probs, dim=-1)
    dirichlet_tokens = config.dirichlet_support[interpolated_choices]
    output_tokens = torch.where(use_dirichlet, dirichlet_tokens, output_tokens)
    # above dirichlet threshold you're ngmi
    if wild:
        # sample from random dirichlet distributed
        sampled_probs = sample_dirichlet(new_emwa_dir.float())
        ood_choices = torch.argmax(sampled_probs, dim=-1)
        ood_tokens = config.dirichlet_support[ood_choices]
    else:
        # sample from the pure tuned logprobs
        support_probs = torch.softmax(tuned_logprobs, dim=-1)
        support_choices = torch.multinomial(support_probs, num_samples=1).squeeze(1)
        ood_tokens = config.dirichlet_support[support_choices]
    # Update output tokens where appropriate
    output_tokens = torch.where(outlier_mask & ~use_dirichlet, ood_tokens, output_tokens)
    # update scaffold entropy rate
    scaffold_ent, scaffold_varent = ent_varent(torch.log(interpolated_probs + EPS_tensor))
    new_emwa_ent_scaffold = (
        config.emwa_ent_scaffold_coeff * scaffold_ent + (1 - config.emwa_ent_scaffold_coeff) * state.emwa_ent_scaffold
    )
    new_emwa_varent_scaffold = (
        config.emwa_varent_scaffold_coeff * scaffold_varent
        + (1 - config.emwa_varent_scaffold_coeff) * state.emwa_varent_scaffold
    )
    # update token cross entropies
    batch_indices = torch.arange(bsz, device=device)
    scaffold_token_logprob = torch.log(interpolated_probs[batch_indices, output_tokens] + EPS_tensor)
    naked_token_logprob = torch.log(naked_log_probs[batch_indices, output_tokens] + EPS_tensor)
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


def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize logits to log probabilities with numerical stability."""
    shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_probs = shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)
    return log_probs


def update_token_cross_entropies(
    state: DSState,
    scaffold_token_logprob: torch.Tensor,
    naked_token_logprob: torch.Tensor,
    config: DSConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        torch.einsum("bi,ij,bj->b", state_ent, config.outlier_threshold.bilinear, state_std)
        + torch.einsum("bi,i->b", state_ent, config.outlier_threshold.linear_state_ent)
        + torch.einsum("bi,i->b", state_std, config.outlier_threshold.linear_state_std)
        + naked_ent * config.outlier_threshold.linear_naked_ent
        + naked_varent * config.outlier_threshold.linear_naked_varent
        + config.outlier_threshold.bias
    )


def update_dirichlet_params(dir_support_logp, state, config):
    kl = kl_divergence(dir_support_logp, state.emwa_logp_dir_supp)
    emwa_logp_coeff = (config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))).unsqueeze(-1)
    new_emwa_logp_dir_sup = emwa_logp_coeff * dir_support_logp + (1 - emwa_logp_coeff) * state.emwa_logp_dir_supp
    new_dir_params, _, _ = fit_dirichlet(new_emwa_logp_dir_sup)
    return new_dir_params, new_emwa_logp_dir_sup, kl
