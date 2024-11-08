import torch
import torch.nn.functional as F


def sample_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """Sample from a Dirichlet distribution."""
    dirichlet_dist = torch.distributions.Dirichlet(alpha)
    return dirichlet_dist.sample()


def dirichlet_log_likelihood_from_logprob(logprobs: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute log probability of probs under Dirichlet(alpha)."""
    return (
        torch.sum((alpha - 1.0) * logprobs, dim=-1)
        - torch.lgamma(torch.sum(alpha, dim=-1))
        + torch.sum(torch.lgamma(alpha), dim=-1)
    )


def dirichlet_expectation(alpha: torch.Tensor) -> torch.Tensor:
    """Compute the expectation E[X|X~Dir(alpha)]"""
    alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)
    return alpha / alpha_sum


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


def halley_update(alpha, target_values):
    """
    Compute the Halley's method update direction.
    Supports batched inputs with batch dimension at axis 0.
    EXACTLY mirrors the non-batch version.
    """
    p1 = torch.special.polygamma(1, alpha)
    p2 = torch.special.polygamma(2, alpha)
    S = torch.sum(alpha, dim=-1, keepdim=True)
    s1 = torch.special.polygamma(1, S)
    s2 = torch.special.polygamma(2, S)
    p1_inv = 1.0 / p1
    sum_p1_inv = torch.sum(p1_inv, dim=-1, keepdim=True)
    denom = 1.0 - s1 * sum_p1_inv
    denom = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)
    coeff = s1 / denom
    error = torch.special.digamma(alpha) - torch.special.digamma(S) - target_values
    temp = p1_inv * error
    sum_temp = torch.sum(temp, dim=-1, keepdim=True)
    J_inv_error = temp + coeff * sum_temp * p1_inv
    sum_J_inv_error = torch.sum(J_inv_error, dim=-1, keepdim=True)
    H_J_inv_error = p2 * J_inv_error - s2 * sum_J_inv_error
    temp2 = p1_inv * H_J_inv_error
    sum_temp2 = torch.sum(temp2, dim=-1, keepdim=True)
    J_inv_H_J_inv_error = temp2 + coeff * sum_temp2 * p1_inv
    return -J_inv_error + 0.5 * J_inv_H_J_inv_error


def fit_dirichlet(
    target_values,
    init_alpha=None,
    initial_lr=1.2,
    decay_alpha=0.1,
    decay_beta=2.0,
    decay_gamma=0.25,
    decay_nu=0.75,
    max_iters=140,
    tol=1e-4,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Estimate Dirichlet parameters (alpha) from target logprobs.
    EXACTLY mirrors the non-batch version.
    """
    batch_shape = target_values.shape[:-1]
    n = target_values.shape[-1]
    min_lr = 1e-8
    target_values = target_values.to(torch.float32)  # for large vocab size needs float64
    if init_alpha is None:
        init_alpha = torch.ones(*batch_shape, n, dtype=torch.float32, device=target_values.device)

    alpha = init_alpha
    converged = torch.zeros(batch_shape, dtype=torch.bool, device=target_values.device)
    error_norm = torch.full(batch_shape, float("inf"), device=target_values.device)
    step = torch.ones(batch_shape, dtype=torch.int32, device=target_values.device)

    for _ in range(max_iters):
        S = torch.sum(alpha, dim=-1, keepdim=True)
        digamma_alpha = torch.special.digamma(alpha)
        psi_S = torch.special.digamma(S)
        error = digamma_alpha - psi_S - target_values
        error_norm = torch.linalg.norm(error, dim=-1)
        new_converged = converged | (error_norm < tol)
        step_float = step.float()
        exp_factor = torch.exp(-decay_alpha * (step_float**decay_nu))
        cos_factor = torch.abs(torch.cos(decay_beta / (step_float**decay_gamma)))
        lr = initial_lr * exp_factor * cos_factor
        lr = torch.maximum(lr, torch.tensor(min_lr, device=lr.device, dtype=lr.dtype))
        delta_alpha = halley_update(alpha, target_values)
        scaled_delta_alpha = lr.unsqueeze(-1) * delta_alpha
        max_delta = 0.5 * alpha
        scaled_delta_alpha = torch.clamp(scaled_delta_alpha, -max_delta, max_delta)
        new_alpha = torch.where(
            new_converged.unsqueeze(-1), alpha, torch.maximum(alpha + scaled_delta_alpha, alpha / 2)
        )
        alpha = new_alpha
        converged = new_converged
        step = step + 1

    final_alpha = alpha.to(dtype)
    final_step = step - 1
    final_converged = converged

    return final_alpha, final_step, final_converged


def ent_grad_hess(logits: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = F.softmax(logits / T.unsqueeze(-1), dim=-1)
    log_p = F.log_softmax(logits / T.unsqueeze(-1), dim=-1)
    mu1 = torch.sum(p * log_p, dim=-1)
    diff = log_p - mu1.unsqueeze(-1)
    mu2 = torch.sum(p * diff**2, dim=-1)
    mu3 = torch.sum(p * diff**3, dim=-1)
    return -mu1, mu2 / T, -(2 * mu3 + 3 * mu2) / (T * T)


def temp_tune(
    logits: torch.Tensor,
    target_ent: torch.Tensor,
    T_init: float = 1.0,
    lr: float = 0.1,
    max_iters: int = 10,
    tol: float = 1e-6,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = logits.shape[0]
    logits = logits.to(torch.float32)
    T = torch.full((batch_size,), T_init, dtype=torch.float32, device=logits.device)
    iters = torch.zeros(batch_size, dtype=torch.int32, device=logits.device)
    converged = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)

    for _ in range(max_iters):
        ent, grad, hess = ent_grad_hess(logits, T)
        error = ent - target_ent
        new_converged = converged | (torch.abs(error) < tol)
        denominator = 2 * grad * grad - error * hess
        halley_step = torch.where(
            torch.abs(denominator) > 1e-8, 2 * error * grad / denominator, torch.full_like(T, float("inf"))
        )
        newton_step = torch.where(torch.abs(grad) > 1e-8, error / grad, torch.full_like(T, float("inf")))
        grad_step = torch.where(error > 0, lr * T, -lr * T)
        delta_T = torch.where(
            torch.abs(grad) < 1e-8, grad_step, torch.where(torch.abs(denominator) < 1e-8, newton_step, halley_step)
        )
        delta_T = torch.clamp(delta_T, -0.5 * T, 0.5 * T)
        new_T = torch.where(new_converged, T, torch.maximum(T - delta_T, T / 2))
        T = new_T
        iters = iters + 1
        converged = new_converged

    final_T = T.to(dtype)
    final_iters = iters
    final_converged = converged
    return final_T, final_iters, final_converged
