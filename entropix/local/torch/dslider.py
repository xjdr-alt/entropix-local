import torch
import torch.nn.functional as F
from typing import NamedTuple, Tuple
from .dslider_config import EPS, MAX_TEMP, MIN_TEMP, DSConfig
from .dslider_utils import *


def kl_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two log probability distributions."""
    p = torch.exp(logp)
    kl_elements = p * (logp - logq)
    kl_elements = torch.where(p > 0, kl_elements, torch.zeros_like(p))
    kl = torch.sum(kl_elements, dim=-1)
    return kl


def ent_varent(logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute entropy and varentropu from log probabilities."""
    p = torch.exp(logp)
    ent = -torch.sum(p * logp, dim=-1)
    diff = logp + ent.unsqueeze(-1)
    varent = torch.sum(p * diff**2, dim=-1)
    return ent, varent


def normalize_logits(logits: torch.Tensor, noise_floor: float) -> torch.Tensor:
    """Normalize logits to log probabilities with numerical stability."""
    shifted = logits - torch.max(logits, dim=-1, keepdim=True)[0]

    # Normalize using log_softmax (equivalent to shifted - logsumexp)
    normalized = torch.log_softmax(shifted + EPS, dim=-1)

    # Apply noise floor
    return torch.where(normalized < noise_floor, torch.log(torch.tensor(EPS)), normalized)


class DSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""

    emwa_dir: torch.Tensor
    emwa_logp_on_supp: torch.Tensor
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

    def to(self, device: torch.device) -> "DSState":
        return DSState(
            emwa_dir=self.emwa_dir.to(device),
            emwa_logp_on_supp=self.emwa_logp_on_supp.to(device),
            emwa_temp=self.emwa_temp.to(device),
            emwa_ent_scaffold=self.emwa_ent_scaffold.to(device),
            emwa_ent_naked=self.emwa_ent_naked.to(device),
            emwa_varent_scaffold=self.emwa_varent_scaffold.to(device),
            emwa_varent_naked=self.emwa_varent_naked.to(device),
            token_cross_ent_scaffold=self.token_cross_ent_scaffold.to(device),
            token_cross_ent_naked=self.token_cross_ent_naked.to(device),
            token_cross_var_scaffold=self.token_cross_var_scaffold.to(device),
            token_cross_var_naked=self.token_cross_var_naked.to(device),
            emwa_dir_ent=self.emwa_dir_ent.to(device),
            emwa_topk_ent_naked=self.emwa_topk_ent_naked.to(device),
        )


def initialize_state(logits: torch.Tensor, bsz: int, config: DSConfig, dtype=torch.float32) -> DSState:
    """Initialize the DSState with specified dtype."""
    _, seqlen, _ = logits.shape
    logprobs = normalize_logits(logits, config.noise_floor)
    ent, varent = ent_varent(logprobs)
    avg_ent, avg_varent = ent.mean(axis=-1), varent.mean(axis=-1)

    topk_logits, topk_indices = torch.topk(logprobs, config.outlier_topk, dim=-1)
    topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
    topk_ent, _ = ent_varent(topk_logprobs)
    avg_topk_ent = topk_ent.mean(dim=-1)

    logprobs_on_supp = normalize_logits(logits[..., config.dirichlet_support], config.noise_floor)
    avg_logprobs_on_supp = torch.mean(logprobs_on_supp, dim=1)

    initial_dir, _, _ = fit_dirichlet(avg_logprobs_on_supp)
    avg_dir_ent = dirichlet_log_likelihood_from_logprob(logprobs_on_supp, initial_dir.unsqueeze(1)).mean(dim=-1)

    topk_token_logprobs = torch.gather(logprobs, -1, topk_indices)
    initial_cross_ent_naked = -topk_token_logprobs.mean(dim=(1, 2))
    initial_cross_var_naked = topk_token_logprobs.var(dim=(1, 2))
    state = DSState(
        emwa_dir=initial_dir.repeat(bsz, 1),
        emwa_logp_on_supp=avg_logprobs_on_supp.repeat(bsz, 1),
        emwa_temp=torch.ones(bsz, dtype=dtype),
        emwa_ent_scaffold=avg_ent.repeat(bsz),
        emwa_ent_naked=avg_ent.repeat(bsz),
        emwa_varent_scaffold=torch.zeros(bsz, dtype=dtype),
        emwa_varent_naked=avg_varent.repeat(bsz),
        token_cross_ent_scaffold=avg_ent.repeat(bsz),
        token_cross_ent_naked=initial_cross_ent_naked.repeat(bsz),
        token_cross_var_scaffold=torch.zeros(bsz, dtype=dtype),
        token_cross_var_naked=initial_cross_var_naked.repeat(bsz),
        emwa_dir_ent=avg_dir_ent.repeat(bsz),
        emwa_topk_ent_naked=avg_topk_ent.repeat(bsz),
    )
    return state


def adaptive_dirichlet_step(
    key: torch.Generator,
    state: DSState,
    logits: torch.Tensor,
    config: DSConfig,
    wild: bool = True,
) -> Tuple[DSState, torch.Tensor]:
    """Single step of the Adaptive Dirichlet Sampler."""
    dtype = logits.dtype
    bsz, vsz = logits.shape
    output_tokens = torch.zeros(bsz, dtype=torch.int32, device=logits.device)
    EPS = torch.tensor(1e-8, dtype=dtype, device=logits.device)

    naked_log_probs = normalize_logits(logits, config.noise_floor)
    # Update naked entropy rate
    naked_ent, naked_varent = ent_varent(naked_log_probs)
    # Fix shape issue!
    new_emwa_ent_naked = update_emwa(naked_ent, state.emwa_ent_naked, config.emwa_ent_naked_coeff)
    new_emwa_varent_naked = update_emwa(naked_varent, state.emwa_varent_naked, config.emwa_varent_naked_coeff)
    # Entropy and varentropy vectors - shape (bsz, 4)
    state_ent = torch.stack(
        [
            state.token_cross_ent_scaffold,
            state.token_cross_ent_naked,
            state.emwa_ent_scaffold,
            state.emwa_ent_naked,
        ],
        dim=1,
    ).float()
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
    state_ent = state_ent.float()
    naked_ent = naked_ent.float()
    naked_varent = naked_varent.float()
    outlier_threshold = compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config)
    outlier_mask = outlier_threshold > 0
    # Update EMWA top-k entropy
    topk_logprobs, topk_indices = torch.topk(naked_log_probs, config.outlier_topk, dim=-1)
    topk_logprobs = normalize_logits(topk_logprobs, config.noise_floor)
    naked_topk_ent, _ = ent_varent(topk_logprobs)
    new_emwa_topk_ent_naked = update_emwa(naked_topk_ent, state.emwa_topk_ent_naked, config.emwa_topk_ent_naked_coeff)
    """
    Argmax policy for concentrated inliers
    """
    argmax_threshold = config.argmax_threshold.weight * state.emwa_topk_ent_naked + config.argmax_threshold.bias
    argmax_mask = (~outlier_mask) & (naked_topk_ent < argmax_threshold)
    argmax_indices = torch.argmax(topk_logprobs, dim=-1)
    argmax_tokens = torch.gather(topk_indices, 1, argmax_indices.unsqueeze(1)).squeeze(1)
    output_tokens = torch.where(argmax_mask, argmax_tokens, output_tokens)
    """
    Top-k temperature tuning policy for dispersed inliers
    """
    inlier_sampling_indices = (~outlier_mask) & (~argmax_mask)
    inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked)
    # adjusted_topk_logprobs = topk_logprobs / inlier_sampling_temp.unsqueeze(1)
    # adjusted_topk_probs = torch.softmax(adjusted_topk_logprobs, dim=-1)
    # sampling_inlier_choices = torch.multinomial(adjusted_topk_probs, num_samples=1, generator=key).squeeze(1)
    sampling_inlier_choices = torch.distributions.Categorical(
        logits=topk_logprobs / inlier_sampling_temp.unsqueeze(1)
    ).sample()
    sampling_inlier_tokens = torch.gather(topk_indices, 1, sampling_inlier_choices.unsqueeze(1)).squeeze(1)
    output_tokens = torch.where(inlier_sampling_indices, sampling_inlier_tokens, output_tokens)
    """
    Tune temperature of outliers to match target entropy
    """
    target_entropy = (
        torch.matmul(state_ent, config.target_entropy.linear)
        + torch.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, dim=-1)
        + config.target_entropy.bias
    )
    temp, _, _ = temp_tune(naked_log_probs.float(), target_entropy)
    new_emwa_temp = update_emwa(temp, state.emwa_temp, config.emwa_temp_coeff)
    tuned_logprobs = normalize_logits(
        naked_log_probs / torch.clamp(temp.unsqueeze(1), MIN_TEMP, MAX_TEMP),
        config.noise_floor,
    )
    """
    Update EMWA log probabilities (on Dirichlet support)
    """
    logprobs_on_supp = normalize_logits(tuned_logprobs[:, config.dirichlet_support], config.noise_floor)
    kl = torch.sum(
        torch.exp(logprobs_on_supp) * (logprobs_on_supp - state.emwa_logp_on_supp),
        dim=-1,
    )
    emwa_logp_coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
    new_emwa_logp_on_supp = update_emwa(logprobs_on_supp, state.emwa_logp_on_supp, emwa_logp_coeff.unsqueeze(-1))
    new_emwa_dir, _, _ = fit_dirichlet(new_emwa_logp_on_supp)
    """
    Update Dirichlet and compute threshold
    """
    dir_log_likelihood = dirichlet_log_likelihood_from_logprob(logprobs_on_supp, state.emwa_dir)
    new_emwa_dir_ent = update_emwa(-dir_log_likelihood, state.emwa_dir_ent, config.emwa_dir_ent_coeff)
    dirichlet_threshold = config.dirichlet_threshold.weight * state.emwa_dir_ent + config.dirichlet_threshold.bias
    use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
    if wild:  # If wild, sample from Dirichlet, else use expectation
        dir_probs = sample_dirichlet(new_emwa_dir.float())
    else:
        dir_probs = dirichlet_expectation(new_emwa_dir.float())
    """
    Below Dirichlet threshold, interpolate and sample (can improve this in the future)
    """
    kl = torch.sum(dir_probs * (torch.log(dir_probs + EPS) - logprobs_on_supp), dim=-1)
    perturb_coeff = 1 - torch.pow(config.perturb_base_coeff, -config.perturb_exp_coeff * (1 / (kl + EPS)))
    interpolated_probs = perturb_coeff.unsqueeze(1) * dir_probs + (1 - perturb_coeff.unsqueeze(1)) * torch.exp(
        logprobs_on_supp
    )
    # In use_dirichlet case, take argmax of the interpolated probabilities
    dirichlet_choices = torch.argmax(interpolated_probs, dim=-1)
    dirichlet_tokens = config.dirichlet_support[dirichlet_choices]
    output_tokens = torch.where(use_dirichlet, dirichlet_tokens, output_tokens)
    """
    Above Dirichlet threshold, sample from Dirichlet
    """
    ood_choices = torch.multinomial(dir_probs, num_samples=1, generator=key).squeeze(1)
    ood_tokens = config.dirichlet_support[ood_choices]
    output_tokens = torch.where(outlier_mask & ~use_dirichlet, ood_tokens, output_tokens)
    # Update scaffold entropy rate
    scaffold_ent, scaffold_varent = ent_varent(torch.log(interpolated_probs + EPS))
    new_emwa_ent_scaffold = update_emwa(scaffold_ent, state.emwa_ent_scaffold, config.emwa_ent_scaffold_coeff)
    new_emwa_varent_scaffold = update_emwa(
        scaffold_varent, state.emwa_varent_scaffold, config.emwa_varent_scaffold_coeff
    )
    # Update token cross entropies
    batch_indices = torch.arange(bsz, device=logits.device)
    scaffold_token_logprob = torch.log(interpolated_probs[batch_indices, output_tokens] + EPS)
    naked_token_logprob = torch.log(naked_log_probs[batch_indices, output_tokens] + EPS)
    (
        new_token_cross_ent_scaffold,
        new_token_cross_ent_naked,
        new_token_cross_var_scaffold,
        new_token_cross_var_naked,
    ) = update_token_cross_entropies(state, scaffold_token_logprob, naked_token_logprob, config)
    # Assemble new state
    new_state = DSState(
        emwa_dir=new_emwa_dir,
        emwa_logp_on_supp=new_emwa_logp_on_supp,
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
    return (
        new_state,
        output_tokens,
        naked_ent,
        naked_varent,
        scaffold_ent,
        scaffold_varent,
        naked_token_logprob,
        scaffold_token_logprob,
    )


def update_token_cross_entropies(
    state: DSState,
    scaffold_token_logprob: torch.Tensor,
    naked_token_logprob: torch.Tensor,
    config: DSConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update token cross entropy statistics."""
    token_cross_ent_naked = (
        config.token_cross_ent_naked_coeff * (-naked_token_logprob)
        + (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
    )
    token_cross_ent_scaffold = (
        config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob)
        + (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
    )
    token_cross_var_naked = (
        config.token_cross_var_naked_coeff * (token_cross_ent_naked - naked_token_logprob) ** 2
        + (1 - config.token_cross_var_naked_coeff) * state.token_cross_var_naked
    )
    token_cross_var_scaffold = (
        config.token_cross_var_scaffold_coeff * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2
        + (1 - config.token_cross_var_scaffold_coeff) * state.token_cross_var_scaffold
    )
    return (
        token_cross_ent_scaffold,
        token_cross_ent_naked,
        token_cross_var_scaffold,
        token_cross_var_naked,
    )


def update_emwa(new: torch.Tensor, old: torch.Tensor, coeff: float | torch.Tensor) -> torch.Tensor:
    return coeff * new + (1 - coeff) * old


def compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config):
    return (
        torch.einsum("bi,ij,bj->b", state_ent, config.outlier_threshold.bilinear, state_std)
        + torch.einsum("bi,i->b", state_ent, config.outlier_threshold.linear_state_ent)
        + torch.einsum("bi,i->b", state_std, config.outlier_threshold.linear_state_std)
        + naked_ent * config.outlier_threshold.linear_naked_ent
        + naked_varent * config.outlier_threshold.linear_naked_varent
        + config.outlier_threshold.bias
    )


def update_dirichlet_params(tuned_logprops_on_supp, state, config):
    kl = kl_divergence(tuned_logprops_on_supp, state.emwa_logp_dir_supp)
    emwa_logp_coeff = (config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))).unsqueeze(-1)
    new_emwa_logp_dir_sup = emwa_logp_coeff * tuned_logprops_on_supp + (1 - emwa_logp_coeff) * state.emwa_logp_dir_supp
    new_dir_params, _, _ = fit_dirichlet(new_emwa_logp_dir_sup)
    return new_dir_params, new_emwa_logp_dir_sup
