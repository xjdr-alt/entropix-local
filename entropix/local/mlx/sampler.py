import mlx.core as mx
from typing import Tuple, Dict

#global inports
from entropix.local.config import SamplerState, SamplerConfig, EntropixConfig

#framework specific improts
from entropix.local.mlx.metrics import calculate_metrics

def multinomial_sample_one(probs_sort: mx.array, rng_key) -> mx.array:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = mx.random.uniform(shape=probs_sort.shape, key=rng_key)
    return mx.argmax(probs_sort / q, axis=-1)[..., None].astype(mx.int32)

def _sample(logits: mx.array, temperature: float, top_p: float, top_k: int, min_p: float, rng_key=None) -> mx.array:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = mx.softmax(logit / temperature, axis=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = mx.max(probs, axis=-1, keepdims=True)
        indices_to_remove = probs < (min_p * p_max)
        inf_mask = mx.zeros_like(logit) - float('inf')
        logit = mx.where(indices_to_remove, inf_mask, logit)
        probs = mx.softmax(logit, axis=-1)

    # Get indices that would sort probs in descending order
    sorted_indices = mx.argsort(-probs, axis=-1)  # Negative for descending order
    probs_sort = mx.take_along_axis(probs, sorted_indices, axis=-1)
    
    # Take top k
    k = min(top_k, probs.shape[-1])
    probs_sort = probs_sort[..., :k]
    sorted_indices = sorted_indices[..., :k]

    # Apply top-p sampling
    probs_sum = mx.cumsum(probs_sort, axis=-1)
    mask = mx.where(probs_sum - probs_sort > top_p, mx.ones_like(probs_sort), mx.zeros_like(probs_sort))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / (mx.sum(probs_sort, axis=-1, keepdims=True) + 1e-10)

    # Sample and map back to original token space
    next_token = multinomial_sample_one(probs_sort, rng_key)
    next_token = mx.take_along_axis(sorted_indices, next_token, axis=-1)
    
    return next_token.astype(mx.int32)


def adaptive_sample(logits: mx.array, temperature: float, epsilon: float = 0.01, rng_key=None) -> mx.array:
    """
    Perform adaptive sampling by dynamically adjusting the candidate set size based on entropy and varentropy.
    """
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = mx.softmax(logit / temperature, axis=-1)

    # Explicitly reverse the arrays
    sorted_indices = mx.argsort(-probs, axis=-1)  # Negative for descending order
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)


    # Initialize candidate set size
    candidate_mask = mx.zeros_like(sorted_probs)
    cumulative_entropy = mx.zeros(bsz)
    cumulative_varentropy = mx.zeros(bsz)
    # Initial entropy calculation
    previous_entropy = -mx.sum(sorted_probs[0] * mx.log2(mx.clip(sorted_probs[0], 1e-10, 1.0)))

    i = 0
    while i < sorted_probs.shape[-1]:
        current_prob = sorted_probs[:, i]

        # Update entropy and varentropy with current token
        current_entropy = -mx.sum(current_prob * mx.log2(mx.clip(current_prob, 1e-10, 1.0)))
        current_varentropy = mx.sum(current_prob * (mx.log2(mx.clip(current_prob, 1e-10, 1.0)) +
                                                  mx.expand_dims(cumulative_entropy, -1))**2)

        entropy_reduction = cumulative_entropy - current_entropy
        varentropy_reduction = cumulative_varentropy - current_varentropy

        # Update mask where entropy reduction is sufficient
        candidate_mask = candidate_mask.at[..., i].set(
            (entropy_reduction >= epsilon).astype(mx.float32)
        )

        # Update cumulative values
        cumulative_entropy = mx.where(entropy_reduction >= epsilon,
                                    cumulative_entropy,
                                    current_entropy)
        cumulative_varentropy = mx.where(entropy_reduction >= epsilon,
                                       cumulative_varentropy,
                                       current_varentropy)

        # Check continuation condition
        if not mx.any(entropy_reduction >= epsilon) or i >= sorted_probs.shape[-1] - 1:
            break

        i += 1

    # Mask out tokens not in the candidate set
    candidate_probs = sorted_probs * candidate_mask
    candidate_probs = candidate_probs / mx.sum(candidate_probs, axis=-1, keepdims=True)

    # Sample from the final candidate set
    next_token = multinomial_sample_one(candidate_probs, rng_key)
    next_token = mx.take_along_axis(sorted_indices, next_token, axis=-1)

    return next_token.astype(mx.int32)

def sample(gen_tokens: mx.array, logits: mx.array, attention_scores: mx.array, cfg: SamplerConfig, entropix_cfg: EntropixConfig,
           clarifying_question_token: int = 2564, rng_key=None) -> Tuple[mx.array, SamplerState]:
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if entropix_cfg.state_flowing and (ent < cfg.low_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold and
        attn_ent < cfg.low_attention_entropy_threshold and
        attn_vent < cfg.low_attention_varentropy_threshold and
        (not entropix_cfg.state_extras_agreement or agreement < cfg.low_agreement_threshold) and
        (not entropix_cfg.state_extras_interaction_strength or interaction_strength < cfg.low_interaction_strength_threshold)):

        sampler_state = SamplerState.FLOWING
        sampled_token = mx.argmax(logits[:, -1], axis=-1, keepdims=True).astype(mx.int32)

        return sampled_token, sampler_state

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif entropix_cfg.state_treading and (ent > cfg.high_logits_entropy_threshold and
          vent < cfg.low_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold and
          (not entropix_cfg.state_extras_agreement or agreement < cfg.low_agreement_threshold) and
          (not entropix_cfg.state_extras_interaction_strength or interaction_strength < cfg.low_interaction_strength_threshold)):
        sampler_state = SamplerState.TREADING
        # Insert a clarifying question token if not already present

        if not mx.any(mx.equal(gen_tokens[:, -1], mx.array([clarifying_question_token]))):

            sampled_token = mx.array([[clarifying_question_token]], dtype=mx.int32)
            return sampled_token, sampler_state
        else:
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent

            sampled_token = _sample(
                logits,
                temperature=min(1.5, cfg.temperature * temp_adj),
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_p,
                rng_key=rng_key
            )
            return sampled_token, sampler_state

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif entropix_cfg.state_exploring and (ent < cfg.high_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          (not entropix_cfg.state_extras_agreement or agreement < cfg.low_agreement_threshold) and
          (not entropix_cfg.state_extras_interaction_strength or interaction_strength > cfg.low_interaction_strength_threshold)):
        sampler_state = SamplerState.EXPLORING
        temp_adj = cfg.low_entropy_interaction_strength_offset + cfg.low_entropy_interaction_strength_coefficient * interaction_strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        sampled_token = _sample(
            logits,
            temperature=min(1.5, cfg.temperature * temp_adj),
            top_p=cfg.top_p,
            top_k=top_k_adj,
            min_p=cfg.min_p,
            rng_key=rng_key
        )
        return sampled_token, sampler_state

    # High Entropy, High Varentropy: "resampling in the mist"
    elif entropix_cfg.state_resampling and (ent > cfg.medium_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent > cfg.high_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          (not entropix_cfg.state_extras_agreement or agreement > cfg.high_agreement_threshold) and
          (not entropix_cfg.state_extras_interaction_strength or interaction_strength > cfg.high_interaction_strength_threshold)):
        sampler_state = SamplerState.RESAMPLING
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attn_vent
        top_p_adj = max(0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attn_ent)
        sampled_token = _sample(
            logits,
            temperature=max(2.0, cfg.temperature * temp_adj),
            top_p=top_p_adj,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            rng_key=rng_key
        )
        return sampled_token, sampler_state

    # All other cases: use adaptive sampling
    else:

        sampler_state = SamplerState.ADAPTIVE
        '''temperature = 0.666
        sampled_token = adaptive_sample(
            logits,
            temperature=temperature,
            epsilon=0.1,
            generator=generator
        )'''
        logits_uncertainty = ent + vent
        attn_uncertainty = attn_ent + attn_vent

        temperature = cfg.temperature * (
            1 +
            cfg.adaptive_temperature_logits_coefficient * ent +
            cfg.adaptive_temperature_attention_coefficient * attn_ent -
            cfg.adaptive_temperature_agreement_coefficient * agreement
        )
        top_p = mx.clip(
            cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attn_vent),
            0.1,
            1.0
        )
        top_k = int(mx.clip(
            mx.round(mx.array(cfg.top_k) * (
                1 +
                cfg.adaptive_top_k_interaction_coefficient * interaction_strength -
                cfg.adaptive_top_k_agreement_coefficient * agreement
            )),
            1,
            100
        ).item())
        min_p = mx.clip(
            cfg.min_p * (1 - cfg.adaptive_min_p_coefficient * vent),
            0.01,
            0.5
        )


        samples = []

        for _ in range(cfg.n_adaptive_samples):
            sample = _sample(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                rng_key=rng_key
            )
            #print(f"Sample {_}: token={sample.tolist()}")  # Convert to Python list
            samples.append(sample)

        def score_sample(sample):
            # Ensure sample is a 1D tensor of indices
            sample_indices = sample.reshape(-1)
            
            # Create one-hot encoding manually
            num_classes = logits.shape[-1]
            one_hot = mx.zeros((sample_indices.shape[0], num_classes))

            for i in range(sample_indices.shape[0]):
                one_hot[i, sample_indices[i]] = 1.0
            
            # Calculate log probability
            log_probs = mx.log(mx.softmax(logits[:, -1], axis=-1))
            log_prob = mx.sum(log_probs * one_hot, axis=-1)

            confidence_score = (
                (1 - ent / cfg.high_logits_entropy_threshold) * cfg.adaptive_score_logits_entropy_coefficient +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.adaptive_score_attention_entropy_coefficient +
                (1 - vent / cfg.high_logits_varentropy_threshold) * cfg.adaptive_score_logits_varentropy_coefficient +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.adaptive_score_attention_varentropy_coefficient +
                (agreement / cfg.high_agreement_threshold) * cfg.adaptive_score_agreement_coefficient +
                (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.adaptive_score_interaction_strength_coefficient
            )
            return log_prob + confidence_score

        sample_scores = mx.stack([score_sample(sample) for sample in samples])
        best_sample_idx = mx.argmax(sample_scores)
        best_idx = best_sample_idx.item()
        sampled_token = samples[best_idx]

        return sampled_token, sampler_state
    