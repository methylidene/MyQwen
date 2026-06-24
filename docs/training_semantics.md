# Training Semantics

## Roles

`policy` is the trainable `ModelBackend`. `reference_policy` is an optional,
frozen backend used only for KL regularization. `old_policy` is **not** a
second mutable model: it is the per-token log probability tensor captured in
the rollout before an optimizer update. It is never recomputed after update.

## SFT

For input tokens `x` and labels `y`, prompt labels and padding are `-100`.
The backend computes cross entropy only over completion labels:

`L_SFT = - sum_t m_t log p_theta(y_t | x_<t) / sum_t m_t`

where `m_t = 1[y_t != -100]`. Gradient accumulation divides each micro-loss
by the configured accumulation count; the final partial accumulation is also
stepped. `train_tokens` counts labels where `m_t=1`.

## GRPO Rollout

For each prompt, the generator produces `G=group_size` completions with the
resolved `temperature`, `top_p`, `max_new_tokens`, prompt formatter and pad
id stored in `config.json`. The completion mask excludes both prompt actions
and padding. Rollout captures:

- `old_logprobs = log p_old(a_t | prompt, a_<t)`;
- optional frozen `reference_logprobs`;
- component reward dictionaries;
- completion lengths, truncation and rollout wall time.

Rewards are normalized only inside one prompt's group:

`A_i = (r_i - mean(r)) / std(r)`.

If group standard deviation is at most `advantage_epsilon`, all advantages
are zero and `zero_variance_groups` is incremented.

## GRPO Update and Clipping

For response tokens only,

`rho_i,t = exp(log p_theta(a_i,t) - old_logprob_i,t)`

`L_policy = - mean_i mean_t min(rho_i,t A_i, clip(rho_i,t,1-eps,1+eps) A_i)`

The optional sampled KL estimator is

`KL = mean_t(exp(ref_logp - policy_logp) - 1 - (ref_logp - policy_logp))`

and the optimized loss is

`L = L_policy + beta_kl * KL - entropy_coef * H`.

No third-party default beta, temperature or chat template is consulted.
`beta_kl`, clipping epsilon, entropy coefficient, generation parameters and
formatter settings are serialized in resolved config/checkpoints.

## Checkpoints

Legacy final model output remains `output_dir/checkpoint`. Rotating trainer
checkpoints are `checkpoint-step-XXXXXXXX/` and contain `model/`, optimizer
state, RNG states, `TrainerState`, and `resolved_config.json`. Resume loads
model from the checkpoint's `model/` directory before restoring optimizer and
RNG state. This supports Hugging Face/PEFT checkpoints and custom backend
checkpoints through the existing `ModelBackend` contract.
