import mlx.core as mx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from functools import partial
import numpy as np

# Configure JAX to use Metal backend
jax.config.update('jax_platform_name', 'METAL')
jax.config.update('jax_enable_x64', False)  # METAL requires float32

# Conversion utilities
def _convert_to_jax(x):
    """Convert MLX array to JAX array."""
    if isinstance(x, mx.array):
        return jnp.asarray(x)  # NumPy conversion happens implicitly
    return x

def _convert_to_mlx(x):
    """Convert JAX array to MLX array."""
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        return mx.array(np.array(x))  # Convert JAX array to NumPy first
    return x

def _convert_tuple_to_mlx(t):
    """Convert tuple of JAX arrays to tuple of MLX arrays."""
    return tuple(_convert_to_mlx(x) for x in t)

# JAX implementations - now running on MPS
@partial(jax.jit, backend='METAL')
def _jax_sample_dirichlet(key, alpha):
    gamma_samples = jax.random.gamma(key, alpha, shape=alpha.shape)
    return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)

@partial(jax.jit, backend='METAL')
def _jax_dirichlet_log_likelihood_from_logprob(logprobs, alpha):
    return (
        jnp.sum((alpha - 1.0) * logprobs, axis=-1)
        - jsp.gammaln(jnp.sum(alpha, axis=-1))
        + jnp.sum(jsp.gammaln(alpha), axis=-1)
    )

@partial(jax.jit, backend='METAL')
def _jax_dirichlet_expectation(alpha):
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    return alpha / alpha_sum

@partial(jax.jit, backend='METAL')
def _jax_dirichlet_expected_entropy(alpha):
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    K = alpha.shape[-1]
    log_beta = jnp.sum(jsp.gammaln(alpha), axis=-1) - jsp.gammaln(alpha_sum.squeeze())
    digamma_sum = jsp.digamma(alpha_sum)
    second_term = (alpha_sum.squeeze() - K) * digamma_sum.squeeze()
    digamma_alpha = jsp.digamma(alpha)
    third_term = -jnp.sum((alpha - 1) * digamma_alpha, axis=-1)
    return log_beta + second_term + third_term

@partial(jax.jit, backend='METAL')
def _jax_dirichlet_expected_varentropy(alpha):
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    expected_x = alpha / alpha_sum
    digamma_alpha = jsp.digamma(alpha)
    trigamma_alpha = jsp.polygamma(1, alpha)
    squared_plus_deriv = digamma_alpha**2 + trigamma_alpha
    return jnp.sum(expected_x * squared_plus_deriv, axis=-1)

@partial(jax.jit, backend='METAL')
def _jax_halley_update(alpha, target_values):
    # Original halley_update implementation
    p1 = jsp.polygamma(1, alpha)
    p2 = jsp.polygamma(2, alpha)
    S = jnp.sum(alpha, axis=-1, keepdims=True)
    s1 = jsp.polygamma(1, S)
    s2 = jsp.polygamma(2, S)
    p1_inv = 1.0 / p1
    sum_p1_inv = jnp.sum(p1_inv, axis=-1, keepdims=True)
    denom = 1.0 - s1 * sum_p1_inv
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
    coeff = s1 / denom
    error = jsp.digamma(alpha) - jsp.digamma(S) - target_values
    temp = p1_inv * error
    sum_temp = jnp.sum(temp, axis=-1, keepdims=True)
    J_inv_error = temp + coeff * sum_temp * p1_inv
    sum_J_inv_error = jnp.sum(J_inv_error, axis=-1, keepdims=True)
    H_J_inv_error = p2 * J_inv_error - s2 * sum_J_inv_error
    temp2 = p1_inv * H_J_inv_error
    sum_temp2 = jnp.sum(temp2, axis=-1, keepdims=True)
    J_inv_H_J_inv_error = temp2 + coeff * sum_temp2 * p1_inv
    return -J_inv_error + 0.5 * J_inv_H_J_inv_error

@partial(jax.jit, backend='METAL')
def _jax_ent_grad_hess(logits, T):
    p = jax.nn.softmax(logits / T[..., None], axis=-1)
    log_p = jax.nn.log_softmax(logits / T[..., None], axis=-1)
    mu1 = jnp.sum(p * log_p, axis=-1)
    diff = log_p - mu1[..., None]
    mu2 = jnp.sum(p * diff**2, axis=-1)
    mu3 = jnp.sum(p * diff**3, axis=-1)
    return -mu1, mu2 / T, -(2 * mu3 + 3 * mu2) / (T * T)

# The fit_dirichlet and temp_tune functions remain largely the same,
# but we'll add the MPS backend specification to their scan operations

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9), backend='METAL')
def _jax_fit_dirichlet(
    target_values,
    init_alpha=None,
    initial_lr=1.2,
    decay_alpha=0.1,
    decay_beta=2.0,
    decay_gamma=0.25,
    decay_nu=0.75,
    max_iters=140,
    tol=1e-4,
    dtype=jnp.float32,
):
    # Original fit_dirichlet implementation
    batch_shape = target_values.shape[:-1]
    n = target_values.shape[-1]
    min_lr = 1e-8
    target_values = target_values.astype(jnp.float32)
    if init_alpha is None:
        init_alpha = jnp.ones((*batch_shape, n), dtype=jnp.float32)

    def scan_body(carry, _):
        alpha, converged, error_norm, step = carry
        S = jnp.sum(alpha, axis=-1, keepdims=True)
        digamma_alpha = jsp.digamma(alpha)
        psi_S = jsp.digamma(S)
        error = digamma_alpha - psi_S - target_values
        error_norm = jnp.linalg.norm(error, axis=-1)
        new_converged = converged | (error_norm < tol)
        exp_factor = jnp.exp(-decay_alpha * (step**decay_nu))
        cos_factor = jnp.abs(jnp.cos(decay_beta / (step**decay_gamma)))
        lr = initial_lr * exp_factor * cos_factor
        lr = jnp.maximum(lr, min_lr)
        delta_alpha = _jax_halley_update(alpha, target_values)
        scaled_delta_alpha = lr[..., None] * delta_alpha
        max_delta = 0.5 * alpha
        scaled_delta_alpha = jnp.clip(scaled_delta_alpha, -max_delta, max_delta)
        new_alpha = jnp.where(
            new_converged[..., None],
            alpha,
            jnp.maximum(alpha + scaled_delta_alpha, alpha / 2),
        )
        return (new_alpha, new_converged, error_norm, step + 1), None

    init_state = (
        init_alpha,
        jnp.zeros(batch_shape, dtype=jnp.bool_),
        jnp.full(batch_shape, jnp.inf),
        jnp.ones(batch_shape, dtype=jnp.int32),
    )
    (final_alpha, final_converged, _, final_step), _ = jax.lax.scan(
        scan_body, init_state, None, length=max_iters
    )

    return final_alpha.astype(dtype), final_step - 1, final_converged

@partial(jax.jit, static_argnums=(3, 4, 5, 6), backend='METAL')
def _jax_temp_tune(
    logits,
    target_ent,
    T_init=1.0,
    lr=0.1,
    max_iters=10,
    tol=1e-6,
    dtype=jnp.float32,
):
    # Original temp_tune implementation
    batch_size = logits.shape[0]
    logits = logits.astype(jnp.float32)

    def scan_body(carry, _):
        T, iters, converged = carry
        ent, grad, hess = _jax_ent_grad_hess(logits, T)
        error = ent - target_ent
        new_converged = converged | (jnp.abs(error) < tol)
        denominator = 2 * grad * grad - error * hess
        halley_step = jnp.where(
            jnp.abs(denominator) > 1e-8,
            2 * error * grad / denominator,
            jnp.full_like(T, jnp.inf),
        )
        newton_step = jnp.where(
            jnp.abs(grad) > 1e-8, error / grad, jnp.full_like(T, jnp.inf)
        )
        grad_step = jnp.where(error > 0, lr * T, -lr * T)

        delta_T = jnp.where(
            jnp.abs(grad) < 1e-8,
            grad_step,
            jnp.where(jnp.abs(denominator) < 1e-8, newton_step, halley_step),
        )
        delta_T = jnp.clip(delta_T, -0.5 * T, 0.5 * T)
        new_T = jnp.where(new_converged, T, jnp.maximum(T - delta_T, T / 2))
        return (new_T, iters + 1, new_converged), None

    init_state = (
        jnp.full((batch_size,), T_init, dtype=jnp.float32),
        jnp.zeros(batch_size, dtype=jnp.int32),
        jnp.zeros(batch_size, dtype=jnp.bool_),
    )
    (final_T, final_iters, final_converged), _ = jax.lax.scan(
        scan_body, init_state, None, length=max_iters
    )
    return final_T.astype(dtype), final_iters, final_converged

# MLX-facing wrapper functions
def sample_dirichlet(rng: int, alpha: mx.array) -> mx.array:
    """MLX wrapper for sampling from a Dirichlet distribution."""
    jax_alpha = _convert_to_jax(alpha)
    key = jax.random.PRNGKey(rng)
    result = _jax_sample_dirichlet(key, jax_alpha)
    return _convert_to_mlx(result)

def dirichlet_log_likelihood_from_logprob(logprobs: mx.array, alpha: mx.array) -> mx.array:
    """MLX wrapper for Dirichlet log likelihood."""
    jax_logprobs = _convert_to_jax(logprobs)
    jax_alpha = _convert_to_jax(alpha)
    result = _jax_dirichlet_log_likelihood_from_logprob(jax_logprobs, jax_alpha)
    return _convert_to_mlx(result)

def dirichlet_expectation(alpha: mx.array) -> mx.array:
    """MLX wrapper for Dirichlet expectation."""
    jax_alpha = _convert_to_jax(alpha)
    result = _jax_dirichlet_expectation(jax_alpha)
    return _convert_to_mlx(result)

def dirichlet_expected_entropy(alpha: mx.array) -> mx.array:
    """MLX wrapper for Dirichlet expected entropy."""
    jax_alpha = _convert_to_jax(alpha)
    result = _jax_dirichlet_expected_entropy(jax_alpha)
    return _convert_to_mlx(result)

def dirichlet_expected_varentropy(alpha: mx.array) -> mx.array:
    """MLX wrapper for Dirichlet expected varentropy."""
    jax_alpha = _convert_to_jax(alpha)
    result = _jax_dirichlet_expected_varentropy(jax_alpha)
    return _convert_to_mlx(result)

def fit_dirichlet(
    target_values: mx.array,
    init_alpha=None,
    initial_lr=1.2,
    decay_alpha=0.1,
    decay_beta=2.0,
    decay_gamma=0.25,
    decay_nu=0.75,
    max_iters=140,
    tol=1e-4,
) -> tuple[mx.array, mx.array, mx.array]:
    """MLX wrapper for fitting Dirichlet parameters."""
    jax_target = _convert_to_jax(target_values)
    jax_init_alpha = _convert_to_jax(init_alpha) if init_alpha is not None else None
    
    result = _jax_fit_dirichlet(
        jax_target,
        jax_init_alpha,
        initial_lr,
        decay_alpha,
        decay_beta,
        decay_gamma,
        decay_nu,
        max_iters,
        tol,
        dtype=jnp.float32,
    )
    
    return _convert_tuple_to_mlx(result)

def temp_tune(
    logits: mx.array,
    target_ent: mx.array,
    T_init: float = 1.0,
    lr: float = 0.1,
    max_iters: int = 10,
    tol: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array]:
    """MLX wrapper for temperature tuning."""
    jax_logits = _convert_to_jax(logits)
    jax_target_ent = _convert_to_jax(target_ent)
    
    result = _jax_temp_tune(
        jax_logits,
        jax_target_ent,
        T_init,
        lr,
        max_iters,
        tol,
        dtype=jnp.float32,
    )
    result_mlx =_convert_tuple_to_mlx(result)
    return result_mlx