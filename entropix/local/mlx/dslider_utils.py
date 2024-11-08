import mlx.core as mx
import mlx.nn as nn
import numpy as np
import scipy.special as sp
from einops.array_api import rearrange
from typing import Tuple

def halley_update(alpha, target_values):
    """
    Computes the Halley's method update direction.
    """

    p1 = mx.array(sp.polygamma(1, alpha))
    p2 = mx.array(sp.polygamma(2, alpha))

    S = mx.sum(alpha, axis = -1, keepdims = True)
    s1 = mx.array(sp.polygamma(1, S))
    s2 = mx.array(sp.polygamma(2, S))

    p1_inv = 1 / p1

    sum_p1_inv = mx.sum(p1_inv, axis = -1, keepdims = True)

    denom = 1.0 - s1 * sum_p1_inv

    denom = mx.where(mx.abs(denom) < 1e-12, 1e-12, denom)

    coeff = s1 / denom

    error = sp.digamma(alpha) - sp.digamma(S) - target_values

    temp = p1_inv * error

    sum_temp = mx.sum(temp, axis = -1, keepdims = True)

    J_inv_error = temp + coeff * sum_temp * p1_inv

    sum_J_inv_error = mx.sum(J_inv_error, axis = -1, keepdims = True)

    H_J_inv_error = p2 * J_inv_error - s2 * sum_J_inv_error

    temp2 = p1_inv * H_J_inv_error

    sum_temp2 = mx.sum(temp2, axis = -1, keepdims = True)

    J_inv_H_J_inv_error = temp2 + coeff * sum_temp2 * p1_inv

    return -J_inv_error + 0.5 * J_inv_H_J_inv_error

def fit_dirichlet(
    target_values,
    init_alpha,
    initial_lr = 1.2,
    decay_alpha = 0.1,
    decay_beta = 2.0,
    decay_gamma = 0.25,
    decay_nu = 0.75,
    max_iters = 140,
    tol = 1e-4,
):
    """
    Estimate dirichlet parameters (alpha) from Target Logprobs
    """

    batch_shape = target_values.shape[:-1]
    n = target_values.shape[-1]
    min_lr = 1e-8

    if init_alpha is None:
        init_alpha = mx.ones_like(target_values)

    alpha = init_alpha
    converged = mx.zeros(batch_shape)
    error_norm = mx.full(batch_shape, mx.inf)
    step = mx.ones(batch_shape)

    for _ in range(max_iters):
        S = mx.sum(alpha, axis = -1, keepdims = True)
        digamma_alpha = sp.digamma(alpha)
        psi_S = sp.digamma(S)
        error = digamma_alpha - psi_S - target_values
        #error_norm = mx.linalg.norm(error, ord=2, axis=-1)

        new_converged = mx.logical_or(converged, (mx.abs(error) < tol))
        step_float = mx.array(step, dtype = mx.float32)
        exp_factor = mx.exp(-decay_alpha * (step_float ** decay_nu))
        cos_factor = mx.abs(mx.cos(decay_beta / (step_float ** decay_gamma)))

        lr = initial_lr * exp_factor * cos_factor
        lr = mx.maximum(lr, mx.array(min_lr))
        delta_alpha = halley_update(alpha, target_values)
        scaled_delta_alpha = rearrange(lr, "v->v 1") * delta_alpha
        max_delta = 0.5 * alpha
        scaled_delta_alpha = mx.clip(scaled_delta_alpha, -max_delta, max_delta)
        new_alpha = mx.where(
            mx.expand_dims(new_converged, axis=-1),
            alpha,
            mx.maximum(alpha + scaled_delta_alpha, alpha / 2)
        )
        alpha = new_alpha
        converged = new_converged
        step += 1

    final_alpha = alpha
    final_step = step - 1
    final_converged = converged

    return final_alpha, final_step, final_converged

def ent_grad_hess(
    logits: mx.array,
    T: mx.array
) -> Tuple[mx.array, mx.array, mx.array]:
    p = mx.softmax(logits / rearrange(T, "v->v 1"), axis = -1)
    log_p = nn.log_softmax(logits / rearrange(T, "v->v 1"), axis = -1)

    mu1 = mx.sum(p * log_p, axis = -1)
    diff = log_p - rearrange(mu1, "v->v 1")
    mu2 = mx.sum(p * diff**2, axis = -1)
    mu3 = mx.sum(p * diff**3, axis = -1)

    return -mu1, mu2/T, -(2* mu3 + 3 * mu2) / T**2

def temp_tune(
    logits: mx.array,
    target_ent: mx.array,
    T_init: mx.array,
    lr: float = 0.1,
    max_iters: int = 10,
    tol: float = 1e-6,
) -> Tuple[mx.array, mx.array, mx.array]:

    batch_size = logits.shape[0]
    T = mx.full((batch_size,), T_init)
    iters = mx.zeros(batch_size)
    converged = mx.zeros(batch_size)

    for _ in range(max_iters):
        ent, grad, hess = ent_grad_hess(logits, T)
        error = ent - target_ent
        new_converged = mx.logical_or(converged, (mx.abs(error) < tol))

        denominator = 2 * grad ** 2 - error * hess
        halley_step = mx.where(
            mx.abs(denominator) > 1e-8, 2 * error * grad / denominator, mx.inf
        )

        newton_step = mx.where(
            mx.abs(grad) > 1e-8, error / grad, mx.inf
        )

        grad_step = mx.where(
            error > 0, lr * T, -lr * T
        )

        delta_T = mx.where(
            mx.abs(grad) < 1e-8, grad_step, mx.where(
                mx.abs(denominator) < 1e-8, newton_step, halley_step
            )
        )

        delta_T = mx.clip(delta_T, -0.5 * T, 0.5 * T)

        new_T = mx.where(new_converged, T, mx.maximum(T - delta_T, T / 2))

        T = new_T
        iters = iters + 1
        converged = new_converged

    final_T = T
    final_iters = iters
    final_converged = converged

    return final_T, final_iters, final_converged