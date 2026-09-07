# Managed Multi-Reward Deployments

Last updated: 09/07/2026

`reward.deployments` manages multiple model-backed rewards in one training
job. It keeps model lifecycle and resource ownership separate from reward
semantics:

- a deployment owns model resources, loading, router access, and lifecycle;
- a reward function turns a rollout output and a model response into a scalar
  score;
- `MultiVisualRewardManager` combines all term scores with a weighted sum.

Use this configuration when a training job needs more than one model-backed
reward, needs explicit native-model placement, or needs an engine-backed reward
to have its own replica topology. Pure rule rewards do not need a deployment.

## Compatibility and boundaries

Named deployments are optional. Existing configurations without
`reward.deployments` retain the existing reward loop behavior, including:

- `reward.reward_model.enable=True` with one `verl.RewardModelManager`;
- `reward.custom_reward_function.path` / `name`;
- legacy multi-reward functions sharing that one reward-model router; and
- legacy accelerator workers enabled by
  `reward.accelerator_workers.enabled` or the deprecated
  `reward.custom_reward_function.use_accelerator` alias.

Do not enable both the legacy model and named deployments in the same job:

```yaml
reward:
  reward_model:
    enable: false
  deployments:
    # Named deployments go here.
```

`reward.reward_model.enable=True` and a non-empty `reward.deployments` are
rejected together. The configuration is backward compatible, but it does not
automatically convert a legacy `reward_model` entry into a named deployment.

## Architecture

```text
Trainer
  -> selects global_pool or reward_pool for Role.RewardModel
  -> OmniRewardLoopManager
       -> MultiRewardModelManager
            -> EngineRewardDeployment[name]
                 -> verl.RewardModelManager
                      -> router + vLLM/vLLM-Omni replicas + wake/sleep
            -> NativeRewardDeployment[name]
                 -> native worker specification
       -> reward-loop workers
            -> EngineRewardExecutor[name]: router address + model name
            -> NativeRewardExecutor[name]: load -> infer -> sleep
       -> MultiVisualRewardManager
            -> reward functions
            -> final_reward = sum(weight * term_score)
```

`MultiVisualRewardManager` is an aggregator, not a third model backend. The
model backends are `engine` and `native`.

| Backend | Model owner | Use it when | Scoring contract |
| --- | --- | --- | --- |
| `engine` | vLLM or vLLM-Omni through `verl.RewardModelManager` | The engine can load the model and serve the required API. | The reward function receives `reward_router_address` and `model_name`; it sends requests and interprets responses. |
| `native` | One local model in each assigned reward worker | The engine does not support the model or a local Transformers/third-party path is required. | The model returns inference outputs; the configured reward function computes scores. |

For example, an OCR VLM normally uses an OpenAI-compatible
`/v1/chat/completions` request and compares the returned text with the target.
An engine-backed PickScore model uses `/v1/embeddings`, then applies the
PickScore formula. `RewardModelManager` does not implement either model-specific
formula; it only owns the engine lifecycle.

## Resource ownership

The trainer still selects the parent pool using the existing setting:

```text
reward.reward_model.enable_resource_pool=false -> global_pool
reward.reward_model.enable_resource_pool=true  -> reward_pool
```

With named deployments, `MultiRewardModelManager` partitions that selected
parent pool into:

```text
parent pool
  -> one disjoint subpool for every engine deployment
  -> one native parent subpool
       -> explicitly assigned native worker bundles
```

An engine deployment reserves
`replicas * tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size`
bundles, or the explicit `n_gpus_per_node * nnodes` allocation when set. Its
engine owns TP, DP, PP, batching, and scheduling.

`placement.devices` for a native deployment means **bundle indices inside the
native subpool**, not host-global CUDA/NPU indices and not TP ranks. Every
listed bundle hosts one complete local model replica. The lists for distinct
native deployments must be disjoint.

For example, if the native subpool has eight bundles:

```yaml
placement:
  devices: [0, 1, 2, 3]
```

creates four full replicas on the first four native bundles. It does not shard
one model over four devices. Native tensor parallelism is not supported.

## Configuration contract

Every named reward term uses:

```yaml
reward:
  reward_functions:
    term_name:
      deployment: deployment_name
      path: pkg://my_package.reward
      name: compute_score
      weight: 1.0
```

The final score is not normalized by total weight:

```text
final_reward = sum(term.weight * term.score)
```

The default weight is `1.0`.

### Native deployment

A native deployment needs `backend=native`, `model_path`, and a non-empty
`placement.devices` list. It has no engine topology fields such as TP, DP, PP,
`rollout`, `replicas`, `n_gpus_per_node`, or `nnodes`.

It may select a built-in adapter or an explicit native model class:

```yaml
deployments:
  pickscore_native:
    backend: native
    adapter: pickscore
    model_path: /models/PickScore_v1
    placement:
      devices: [0, 1, 2, 3]
    executor:
      kwargs:
        processor_path: /models/PickScore_v1

  custom_native:
    backend: native
    model_path: /models/custom_reward
    placement:
      devices: [4, 5]
    executor:
      model: my_package.reward:CustomNativeModel
      kwargs:
        threshold: 0.5
```

`adapter: pickscore` resolves to
`verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel`. Native
PickScore preserves the existing local burst batching policy. The native
executor waits for active inference before closing the model and clearing the
accelerator cache.

The native executor exposes only an `infer()` handle. Native terms must set
`path` and `name`; that reward function consumes the model output and owns all
score semantics.

### Engine deployment

An engine deployment needs a model path, resource allocation, and rollout
configuration. Every engine term must set `deployment`, `path`, and `name`:

```yaml
deployments:
  ocr:
    backend: engine
    model_path: Qwen/Qwen3-VL-8B-Instruct
    n_gpus_per_node: 2
    nnodes: 1
    rollout:
      name: vllm
      tensor_model_parallel_size: 2
      data_parallel_size: 1
      pipeline_model_parallel_size: 1

reward_functions:
  ocr:
    deployment: ocr
    path: pkg://verl_omni.utils.reward_score.genrm_ocr
    name: compute_score_ocr
    weight: 1.0
```

For this OCR example, the engine deployment loads Qwen3-VL and manages its
replicas. `compute_score_ocr` creates the image-plus-prompt chat request,
submits it to that deployment's router, then turns the transcription into an
OCR score.

An engine deployment does not use an `adapter` field. Keep model-specific
request construction and score computation in the reward function module.

## PickScore examples

### Native PickScore

Use native PickScore when vLLM/vLLM-Omni cannot serve the target checkpoint or
when the local Transformers implementation is required:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: false
  deployments:
    pickscore:
      backend: native
      adapter: pickscore
      model_path: /models/PickScore_v1
      placement:
        devices: [0, 1, 2, 3]
      executor:
        kwargs:
          processor_path: /models/PickScore_v1
  reward_functions:
    pickscore:
      deployment: pickscore
      path: pkg://verl_omni.utils.reward_score.pickscore_reward
      name: compute_score_pickscore_native
      weight: 1.0
```

### Engine PickScore

Use engine PickScore only when the selected vLLM/vLLM-Omni backend supports the
checkpoint as a CLIP pooling model and its embedding endpoint accepts the
configured image input format:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: true
  deployments:
    pickscore_engine:
      backend: engine
      model_path: /models/PickScore_v1
      n_gpus_per_node: 2
      nnodes: 1
      rollout:
        name: vllm
        tensor_model_parallel_size: 1
        data_parallel_size: 1
        pipeline_model_parallel_size: 1
        max_model_len: 77
        max_num_seqs: 8
        limit_images: 1
        enforce_eager: true
        engine_kwargs:
          vllm:
            runner: pooling
  reward_functions:
    pickscore:
      deployment: pickscore_engine
      path: pkg://verl_omni.utils.reward_score.pickscore_reward
      name: compute_score_pickscore_engine
      logit_scale: 98.86447
      weight: 1.0
```

`compute_score_pickscore_engine` requests text and image embeddings through the
deployment router, then computes:

```text
PickScore = logit_scale * cosine(text_embedding, image_embedding) / 26
```

The configured `logit_scale` is the already-exponentiated checkpoint value.
Do not apply `exp()` to it again. A model path alone is not enough: a new
engine-backed model also needs a reward function that defines how to turn that
model's API response into a score.

### One engine and one native model

The following shape uses eight parent-pool bundles for the engine model and
eight for native PickScore. The native values are relative to the native
eight-bundle subpool:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: false
  deployments:
    ocr:
      backend: engine
      model_path: Qwen/Qwen3-VL-8B-Instruct
      n_gpus_per_node: 8
      nnodes: 1
      rollout:
        name: vllm
        tensor_model_parallel_size: 1
    pickscore:
      backend: native
      adapter: pickscore
      model_path: /models/PickScore_v1
      placement:
        devices: [0, 1, 2, 3, 4, 5, 6, 7]
      executor:
        kwargs:
          processor_path: /models/PickScore_v1
  reward_functions:
    ocr:
      deployment: ocr
      path: pkg://verl_omni.utils.reward_score.genrm_ocr
      name: compute_score_ocr
      weight: 0.5
    pickscore:
      deployment: pickscore
      path: pkg://verl_omni.utils.reward_score.pickscore_reward
      name: compute_score_pickscore_native
      weight: 0.5
```

The provided NPU launcher
`examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora_v1_npu_engine_native_test.sh`
is a standalone Qwen-Image-Edit test recipe with this resource shape. It keeps
the base training parameters unchanged, creates eight engine bundles with
TP=1, and assigns eight native PickScore replicas. Set
`PICKSCORE_MODEL_PATH` and, when required, `PICKSCORE_PROCESSOR_PATH` to local
checkpoint paths before launching it.

## Runtime flow

For each reward batch:

```text
1. wake every engine deployment
2. score rule and engine terms in the shared reward worker group
3. score each native deployment in its assigned worker group
4. merge per-term scores and emit one reward/combined value
5. sleep engine deployments
6. native executors close models after their batch and release cache
```

Named deployment jobs use ordinary batch reward computation. The existing
streaming reward interface accepts only one worker list and cannot safely fan a
streaming request across independent native worker groups.

Errors are fail-fast: a reward-loading or scoring exception aborts the training
step. No named reward term silently becomes a zero score.

## Migrate an existing configuration

Keep an existing recipe unchanged when it has one shared engine reward model or
one legacy custom reward. Move to named deployments only when independent model
lifecycle, topology, or native placement is needed.

To migrate a legacy engine-backed OCR reward:

1. Set `reward.reward_model.enable=false` while retaining
   `enable_resource_pool` as the parent-pool selection.
2. Copy the old `model_path` and engine rollout fields into one
   `reward.deployments.<name>` entry.
3. Move the old custom reward function `path/name` into one
   `reward.reward_functions.<term>` entry and bind it with
   `deployment: <name>`.
4. Set the reward-manager configuration to `MultiVisualRewardManager` only
   when the recipe does not already select it; named deployments do this during
   reward-loop initialization.

To migrate a legacy local model, define `backend: native`, use a native model
class or supported adapter, assign non-overlapping native placement bundle
indices, and configure the matching reward term's `path/name`. The model class
returns inference outputs; the reward function computes the score.

Do not put the legacy `reward_model` and its named replacement in the same
configuration. The reward loop rejects that mixture before allocating models.

## Current limits

- Native models are replicated, not tensor parallel.
- A new engine-supported model still needs a reward function for its API and
  score semantics; it is not automatically a complete reward implementation.
- Named engine and native deployments require a parent resource pool large
  enough for their disjoint allocations.
- Engine compatibility must be validated for the selected model, engine
  version, and hardware backend. In particular, a CUDA pooling result does not
  establish NPU pooling support.
