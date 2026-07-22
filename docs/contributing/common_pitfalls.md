# Common Pitfalls

Last updated: 07/22/2026.

---

## Float32 Precision Loss in Stored Rollout Latents

(symptom-float32)=
### Symptom

Training metrics show a systematic negative bias **at step 1** before any weight
update:

- `actor/ratio_mean` consistently below `1.0`, for example `0.99996`;
- `actor/ppo_kl` and `actor/pg_clipfrac` inflated at step 1;
- `actor/pg_clipfrac_higher` is zero, with clipping concentrated on the lower
  side;
- the issue is most visible with rollout correction, but stored trajectory
  precision is also degraded in standard training.

(root-cause-float32)=
### Root cause

`FlowMatchSDEDiscreteScheduler.step()` computes `log_prob` using float32
`prev_sample`, but a lower-precision return path can truncate the latent before
it is stored. The trainer later recomputes the transition log probability from
the stored trajectory in float32, creating a mismatch.

(fix-float32)=
### Fix

The rollout path must preserve float32 at the scheduler boundary:

1. Cast model output to float32 before `scheduler.step()`.
2. Keep the scheduler's returned latent in float32.
3. Store all trajectory latents in float32.
4. Cast latents to the transformer dtype only for the transformer forward pass.

The training adapter is unchanged because it already recomputes transitions in
float32.

(verification-float32)=
### Verification

After the fix, non-bypass training should have `ratio_mean ≈ 1.0` at step 1.
When rollout and training use different attention kernels, a small residual
difference may remain.

| Metric | Before fix | After fp32 fix |
|---|---|---|
| `actor/ratio_mean` | Systematically below `1.0` | Approximately `1.0` |
| `actor/ppo_kl` | Inflated at step 1 | Near numerical baseline |
| `actor/pg_clipfrac` | Excessive lower-side clipping | Substantially reduced |

---

## RoPE Sequence Length Mismatch

(symptom-rope)=
### Symptom

With `step_execution=True`, `actor/ppo_kl` is elevated at step 1 compared with
the full-forward path, even after latent precision is fixed.

The same failure can occur in any pipeline that derives text RoPE length from
the number of non-padding tokens rather than from the padded encoder-hidden-state
width. It is not inherently limited to step execution.

(root-cause-rope)=
### Root cause

Qwen-Image derives text RoPE length from
`encoder_hidden_states.shape[1]`. Under continuous batching, requests may be
padded to a shared embedding width.

Using:

```python
mask.sum()
```

instead produces the valid-token count. Two requests can therefore have
identically padded embedding widths but different RoPE table lengths.

For example, a request with 200 valid tokens padded to width 1058 must use 1058
as its text sequence length for RoPE construction, not 200.

(fix-rope)=
### Fix

Use the padded prompt-embedding width.

Preferred implementation:

```python
from vllm_omni.diffusion.models.qwen_image.rope_utils import (
    txt_seq_lens_from_embeds,
)

txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)
negative_txt_seq_lens = txt_seq_lens_from_embeds(
    negative_prompt_embeds
)
```

Equivalent explicit implementation:

```python
seq_len = int(prompt_embeds.shape[1])
batch_size = int(prompt_embeds.shape[0])
txt_seq_lens = [seq_len] * batch_size
```

Do not derive this value from `prompt_embeds_mask.sum()`.

The integrated Qwen-Image FlowGRPO adapter applies this rule in
`verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`.

(verification-rope)=
### Verification

Compare identical runs with `step_execution=False` and
`step_execution=True`. Prompt embeddings, prompt masks, and text RoPE lengths
must match. The remaining first-step KL difference should be within the expected
numerical tolerance of the selected inference and training kernels.

---

## Float32 Precision Loss in the Step-Execution Scheduler

(symptom-fp32-stepwise)=
### Symptom

When `step_execution=True`:

- `actor/ratio_mean` is consistently below `1.0`;
- `actor/ppo_kl` and `actor/pg_clipfrac` are inflated;
- continuous batching can fail with a mixed-latent-dtype error;
- the same configuration works with `step_execution=False`.

(root-cause-fp32-stepwise)=
### Root cause

A newly admitted request starts with float32 latents. If
`step_scheduler()` stores previously stepped requests in bf16, the engine later
tries to gather requests with different live latent dtypes into one batch.

Storing bf16 trajectory latents also creates a mismatch when training
recomputes log probabilities in float32.

(fix-fp32-stepwise)=
### Fix

Keep both the live request state and the recorded trajectory in float32:

```python
new_latents, log_prob, _, _ = state.scheduler.step(
    noise_pred.to(torch.float32),
    timestep,
    state.latents.to(torch.float32),
    generator=state.sampling.generator,
    noise_level=current_noise_level,
    sde_type=state.sde_type,
    return_logprobs=state.logprobs,
    return_dict=False,
)

state.all_latents.append(new_latents.to(torch.float32))
state.latents = new_latents.to(torch.float32)
```

`denoise_step()` should perform the temporary model-dtype cast:

```python
model_input = input_batch.latents.to(
    self.transformer.img_in.weight.dtype
)
```

(verification-fp32-stepwise)=
### Verification

Run concurrent requests with different arrival times and prompt lengths.
Confirm that:

- live request latents always share one dtype;
- `all_latents.dtype` is float32;
- `ratio_mean ≈ 1.0` at step 1;
- the step-execution and full-forward trajectories agree within tolerance.

---

## Missing or Incomplete Step-Execution Output Contract

(symptom-step-output)=
### Symptom

Generation finishes, but training fails while constructing or padding the
trajectory batch. Typical failures include:

- missing `all_latents`, `all_log_probs`, or `all_timesteps`;
- `prompt_embeds_mask` becoming a non-tensor container;
- a downstream `mask.shape` access failing;
- output tensors remaining on the accelerator in the HTTP-server process.

(root-cause-step-output)=
### Root cause

The upstream model's default `post_decode()` returns the generated sample but
does not know the RL algorithm's trajectory contract.

The step-execution adapter must explicitly package the same fields produced by
the full-forward `forward()` implementation.

(fix-step-output)=
### Fix

Populate every required key, including optional negative-prompt keys:

```python
from dataclasses import replace

return replace(
    output,
    custom_output={
        "all_latents": stacked_latents,
        "all_log_probs": stacked_log_probs,
        "all_timesteps": stacked_timesteps,
        "prompt_embeds": state.prompt_embeds,
        "prompt_embeds_mask": state.prompt_embeds_mask,
        "negative_prompt_embeds":
            state.negative_prompt_embeds,
        "negative_prompt_embeds_mask":
            state.negative_prompt_embeds_mask,
    },
    to_cpu=True,
)
```

Keep the negative-prompt keys even when CFG is disabled and their values are
`None`.

(verification-step-output)=
### Verification

The real-engine regression test should assert:

```text
all_latents.shape[0] = all_timesteps.shape[0] + 1
all_log_probs.shape[0] = all_timesteps.shape[0]
prompt_embeds.shape[:-1] = prompt_embeds_mask.shape
```

It should also verify that required tensors are non-empty and located on CPU
after server-side conversion.

---

## SDE Window: Per-Request vs Group-Consistent Selection

(symptom-sde-window)=
### Symptom

- reward standard deviation grows instead of shrinking;
- reward mean declines or oscillates;
- actor KL and loss become unstable;
- different rollout ranks use different active denoising windows for the same
  training step;
- MixGRPO behaves differently between full-forward and step execution.

(root-cause-sde-window)=
### Root cause

The active SDE window controls where stochastic exploration is injected.

If each request chooses a window independently, reward variation combines:

1. intended exploration noise;
2. unintended variation caused only by using different timestep ranges.

For group-relative optimisation, rollouts being compared should use a
consistent window policy for the same training step.

A second failure occurs when the window is initialised only in `forward()`.
Step execution does not call `forward()`, so the algorithm-specific window setup
is skipped.

(fix-sde-window)=
### Fix

Use an explicit deterministic seed rather than request IDs or process-local
environment state.

For MixGRPO's seeded random strategy:

```python
rng = random.Random(
    int(sde_window_seed) + int(global_steps)
)
start = rng.randint(envelope_start, max_start)
sde_window_range = [start, start + sde_window_size]
```

This allows all rollout ranks to derive the same window for one training step
while changing it across steps.

For the progressive strategy, derive the start from:

```text
global_steps
sde_window_size
iters_per_group
```

Apply the same helper before both execution paths:

```python
def prepare_encode(self, state, **kwargs):
    self._maybe_make_progressive_window(
        state.sampling.extra_args,
        kwargs,
    )
    return super().prepare_encode(state, **kwargs)


def forward(self, req, **kwargs):
    self._maybe_make_progressive_window(
        req.sampling_params.extra_args,
        kwargs,
    )
    return super().forward(req, **kwargs)
```

Do not create independent request-level windows when the algorithm assumes a
shared group window.

(verification-sde-window)=
### Verification

For one training step, log the selected window on every rollout rank and confirm
that it is identical. Across training steps:

- `random` should change deterministically according to
  `sde_window_seed + global_steps`;
- `progressive` should advance according to `iters_per_group`;
- `step_execution=False` and `step_execution=True` should select the same
  window.

---

## Incorrect `_stepwise` Algorithm Registration

(symptom-stepwise-registration)=
### Symptom

- configurations require `flow_grpo_stepwise` or `mix_grpo_stepwise`;
- the async server mutates `model_config.algorithm`;
- normal and step-execution adapters drift apart;
- deleted `verl_omni/experimental` imports break startup.

(root-cause-stepwise-registration)=
### Root cause

Older integrations treated step execution as a separate algorithm registration.
Execution mode and optimisation algorithm are different concerns.

(fix-stepwise-registration)=
### Fix

Keep one normal registration:

```python
@VllmOmniPipelineBase.register(
    "QwenImagePipeline",
    algorithm="flow_grpo",
)
```

Implement `prepare_encode`, `denoise_step`, `step_scheduler`, and `post_decode`
on that adapter. Select execution mode only through:

```yaml
actor_rollout_ref:
  rollout:
    step_execution: true
```

Do not create an experimental package or add a `_stepwise` suffix.

(verification-stepwise-registration)=
### Verification

Both configurations must resolve the same `(architecture, algorithm)` adapter:

```text
step_execution=False -> forward()
step_execution=True  -> step-execution lifecycle
```
