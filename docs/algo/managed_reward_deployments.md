# Managed Multi-Reward Deployments

Last updated: 09/09/2026

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
                 -> native workers + wake/sleep RPC
       -> reward-loop workers
            -> EngineRewardExecutor[name]: router address + model name
            -> NativeRewardExecutor[name]: wake/load -> infer -> optional sleep
       -> MultiVisualRewardManager
            -> reward functions
            -> final_reward = sum(weight * term_score)
```

`MultiVisualRewardManager` is an aggregator, not a third model backend. The
model backends are `engine` and `native`.

Every deployment accepts the same `offload` lifecycle setting:

```yaml
offload: true   # wake before scoring and sleep after scoring (default)
offload: false  # keep the model resident between scoring phases
```

For an engine deployment this controls vLLM/vLLM-Omni sleep mode. For a
native deployment, `false` loads the Transformers or third-party model on its
first scoring phase and keeps that worker-local model alive. The trainer always
uses the same deployment `wake_up()` / `sleep()` contract and does not branch
on the backend.

Keeping a deployment resident requires enough accelerator memory for it to
coexist with the actor rollout. When deployments share `global_pool`, prefer
the default `offload: true` unless the combined resident footprint is known to
fit; otherwise use a separate reward pool.

Choose one backend for a normal single-model reward job:

- use `engine` when vLLM or vLLM-Omni can serve the model and the configured
  reward function can consume its API response;
- use `native` when the engine does not support the model or inference needs a
  Transformers or third-party implementation;
- configure both only when the job genuinely needs multiple model-backed
  reward terms, or when comparing the two inference paths. Running the same
  model through both backends normally duplicates inference work.

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

With named deployments, `MultiRewardModelManager` gives every deployment a
disjoint allocation from that selected parent pool:

```text
parent pool
  -> engine deployment A allocation
  -> engine deployment B allocation
  -> native deployment C allocation
  -> native deployment D allocation
```

An engine deployment reserves
`replicas * tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size`
bundles, or the explicit `n_gpus_per_node * nnodes` allocation when set. Its
engine owns TP, DP, PP, batching, and scheduling.

Internally, every engine allocation is a dedicated `SubRayResourcePool`.
Native allocations are logically at the same level and cannot overlap,
although the implementation maps all native allocations through one internal
native subpool and then selects each deployment's bundles.

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

Resource-pool splitting is ordered. Engine allocations are taken first in
deployment configuration order. The internal native subpool comes next and is
large enough to include its highest configured native bundle index. Any
remaining parent bundles form a trailing allocation unused by named reward
deployments.

For example, on a one-node 16-bundle parent pool:

```bash
ENGINE_REWARD_NPUS=4
NATIVE_REWARD_DEVICES="[0,1,2,3]"
```

produces:

```text
parent bundles 0-3   -> engine reward allocation
parent bundles 4-7   -> native subpool
  native indices 0-3 -> parent bundles 4-7
parent bundles 8-15  -> unused by named reward deployments
```

The native indices above therefore do not select physical NPU 0-3. On a
single node with an unchanged visible-device order, Ray will commonly map the
first two allocations to physical devices 0-3 and 4-7, respectively, but code
must not rely on that physical numbering. Ray and
`ASCEND_RT_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` determine the final
worker-visible device mapping.

When `enable_resource_pool=false`, the parent is `global_pool`. The
actor/rollout still occupies all 16 bundles in this example; reward deployments
are colocated on their eight assigned bundles and use lifecycle offloading to
share those devices. Bundles 8-15 are unused only by named reward deployments,
not by the actor/rollout. With `enable_resource_pool=true`, the same split is
instead applied to the dedicated `reward_pool`.

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
      required: true
```

The final score is not normalized by total weight:

```text
final_reward = sum(term.weight * term.score)
```

The default weight is `1.0`. `required` retains the existing
multi-reward failure policy:

- `required: true`: propagate a scoring failure as fatal and stop the
  training step.
- `required: false` or omitted: record the error and let that term contribute
  zero to the weighted sum.

`required` controls failure handling only. It is not passed to the configured
reward function and does not change model inference or score calculation.

### Native deployment

A native deployment needs `backend=native`, `model_path`, and a non-empty
`placement.devices` list. It has no engine topology fields such as TP, DP, PP,
`rollout`, `replicas`, `n_gpus_per_node`, or `nnodes`.

It may select a built-in adapter or an explicit native model class:

```yaml
deployments:
  pickscore_native:
    backend: native
    offload: true
    adapter: pickscore
    model_path: /models/PickScore_v1
    placement:
      devices: [0, 1, 2, 3]

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
accelerator cache. PickScore-specific construction details such as its CLIP
processor stay in `verl_omni.utils.reward_score.pickscore_reward`; the
deployment supplies only the reward model path.

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
    offload: true
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
        # Example budget for a small pooling model. Tune for the hardware and
        # model instead of copying this value blindly.
        gpu_memory_utilization: 0.1
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

`rollout.gpu_memory_utilization` is the engine's requested accelerator-memory
budget, not the checkpoint size. vLLM may use the budget remaining after model
weights and profiling for cache allocation. For example, `0.5` requests about
32.5 GiB on a 65 GiB device even if the checkpoint weights are only 4 GiB.
This matters most with `offload: false`, because that engine allocation remains
resident while the actor rollout uses the same device. Lower the value only as
far as the reward model, its activation peak, and its required cache capacity
permit. `0.1` above is a PickScore-oriented example, not a universal default.

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
`PICKSCORE_MODEL_PATH` to a local checkpoint path when running offline. The
script is a multi-deployment configuration and parity example, not a
recommendation to score every production sample through PickScore twice.

## Runtime flow

For each reward batch:

```text
1. wake every deployment; native models load in their assigned workers
2. score rule and engine terms in the shared reward worker group
3. score each native deployment in its assigned worker group
4. merge per-term scores and emit one reward/combined value
5. call `sleep` on every deployment in reverse order
```

With `offload: true`, engine deployments release their managed model/cache
memory and native executors close their model objects after scoring. With
`offload: false`, those sleep calls are intentional no-ops and the models stay
resident. Engine services are created eagerly during deployment setup; native
model objects are created in their Ray workers on the first wake so device
binding is correct. This bootstrap difference does not change the common
steady-state `offload` contract.

Named deployment jobs use ordinary batch reward computation. The existing
streaming reward interface accepts only one worker list and cannot safely fan a
streaming request across independent native worker groups.

Each term keeps its configured `required` failure policy. A required term
propagates a reward-function or model-inference failure and stops the training
step. An optional term records `reward/<term>/errors=1` and contributes zero.
Deployment setup and lifecycle errors, such as an unknown deployment, missing
executor, or model wake/load failure, remain fatal because no individual term
can safely recover from them.

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

For example, migrate this legacy engine configuration:

```yaml
reward:
  reward_model:
    enable: true
    enable_resource_pool: false
    model_path: Qwen/Qwen3-VL-8B-Instruct
    rollout:
      name: vllm
      tensor_model_parallel_size: 2
  custom_reward_function:
    path: pkg://verl_omni.utils.reward_score.genrm_ocr
    name: compute_score_ocr
```

to one named engine deployment and one explicitly bound reward term:

```yaml
reward:
  reward_model:
    enable: false
    # This still selects global_pool. Set true for a dedicated reward_pool.
    enable_resource_pool: false
  deployments:
    ocr:
      backend: engine
      offload: true
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

`reward.num_workers` and unrelated trainer settings do not move during this
migration. `reward.reward_model.enable_resource_pool` continues to select the
parent resource pool even though `reward.reward_model.enable` is false. Do not
copy model-specific request or scoring logic into the deployment: keep it in
the configured reward function.

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
