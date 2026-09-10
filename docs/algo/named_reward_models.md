# Named Reward Models

`reward.models` lets one training job use multiple independently managed
model-backed rewards. The framework keeps model inference separate from reward
calculation:

- a named model owns resources, inference access, and lifecycle;
- a reward function consumes model output and computes the score;
- `MultiVisualRewardManager` combines the configured scores by weighted sum.

PickScore is the built-in native example, not the only model supported by the
framework.

## Current scope

Two backends are available:

| Backend | Model owner | Reward-function input |
| --- | --- | --- |
| `engine` | The existing `verl.RewardModelManager` and its vLLM replicas | `reward_router_address` and `model_name` |
| `native` | One full model replica in each assigned reward worker | A `reward_model` inference handle |

The engine path documented and tested here is vLLM. vLLM-Omni reward serving is
not implemented by this change; vLLM-Omni may still be used independently for
the actor rollout.

Existing jobs that do not configure `reward.models` keep their current reward
path. Do not combine `reward.reward_model.enable=true` with named models.
Migration of the existing reward implementations and reward managers will be
designed in a follow-up RFC.

## Configuration

A reward term uses a same-name model automatically:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: false

  models:
    pickscore:
      backend: native
      reward_name: pickscore
      offload: true
      model_path: /models/PickScore_v1
      placement:
        devices: [0, 1, 2, 3]

  reward_functions:
    pickscore:
      path: pkg://verl_omni.utils.reward_score.pickscore_reward
      name: compute_score_pickscore_native
      weight: 1.0
      required: true
```

Use `model` only when the reward term and model have different names, or when
multiple terms share one model:

```yaml
reward_functions:
  semantic_quality:
    model: pickscore
    path: pkg://my_package.rewards
    name: compute_semantic_quality
```

`reward_name` selects a built-in native implementation. A custom native model
uses `executor.model` instead.

Every term retains the existing aggregation contract:

```text
final_reward = sum(term.weight * term.score)
```

`required=true` propagates a scoring failure. An optional term records an error
and contributes zero. Model setup and lifecycle failures remain fatal.

## Resource placement

The existing parent-pool switch is unchanged:

```text
reward.reward_model.enable_resource_pool=false -> global_pool
reward.reward_model.enable_resource_pool=true  -> reward_pool
```

`MultiRewardModelManager` splits that parent pool into disjoint allocations for
the named models. Engine allocations are created first in configuration order,
followed by one native subpool.

For native models, `placement.devices` contains bundle indices relative to the
native subpool. They are not physical CUDA/NPU IDs and are not tensor-parallel
ranks. Each index starts one complete model replica, and indices cannot overlap
between native models.

Native batches are currently padded and split evenly across their assigned
workers. There is no dynamic load balancing, work stealing, or per-request
routing in the native backend.

## Lifecycle and async API

`offload` has the same meaning for both backends:

- `true` (default): wake before scoring and sleep afterward;
- `false`: keep the model resident across training steps.

Independent named models are woken, scored, and slept concurrently. The reward
loop exposes `async_compute_rm_score()` for asynchronous callers and retains
`compute_rm_score()` as a synchronous compatibility entrypoint for current
trainers. Cleanup is attempted even when inference or scoring fails.

Named models still use post-rollout batch reward computation. The current
streaming reward interface accepts one worker list and cannot safely fan one
request out to independent native worker groups. Full streaming admission,
backpressure, and drain semantics are future work.

## Engine model example

```yaml
reward:
  models:
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
      path: pkg://verl_omni.utils.reward_score.genrm_ocr
      name: compute_score_ocr
      weight: 1.0
```

The engine owns serving, batching, and TP/DP/PP. The reward function constructs
the request and turns the response into a score.

## Add a custom native reward model

A native model class owns inference only. It must accept the configured
constructor arguments and expose `infer()`; `close()` is optional.

```python
class MyRewardModel:
    def __init__(self, model_path: str, device, threshold: float = 0.5):
        self.model = load_model(model_path, device=device)
        self.threshold = threshold

    async def infer(self, prompts, images):
        return await run_model(self.model, prompts, images)

    async def close(self):
        await close_model(self.model)
```

The matching reward function receives the inference handle and owns scoring:

```python
async def compute_score(
    data_source,
    solution_image,
    ground_truth,
    extra_info,
    reward_model,
):
    output = await reward_model.infer(
        prompts=[ground_truth or ""],
        images=[solution_image],
    )
    return {"score": float(output[0])}
```

Configure the class and score function separately:

```yaml
reward:
  models:
    quality:
      backend: native
      model_path: /models/my-reward-model
      placement:
        devices: [0, 1]
      executor:
        model: my_package.reward_model:MyRewardModel
        kwargs:
          threshold: 0.6

  reward_functions:
    quality:
      path: pkg://my_package.reward_score
      name: compute_score
```

Synchronous `infer()` and `close()` methods are also accepted and run outside
the reward worker's event loop.

## PickScore validation recipe

The standard Qwen-Image-Edit launcher uses native PickScore. A mixed vLLM and
native PickScore parity recipe is kept under
`tests/special_e2e/run_qwen_image_edit_lora_v1_npu_engine_native.sh` because it
runs the same reward twice for validation and is not a production example.

Engine PickScore uses vLLM's pooling runner and `/v1/embeddings`. Its reward
function computes:

```text
PickScore = logit_scale * cosine(text_embedding, image_embedding) / 26
```

The configured `logit_scale` is already exponentiated and must not be passed
through `exp()` again.

## Current limitations

- Named-model aggregation currently uses the visual reward manager contract.
- Native models are replicated; FSDP and tensor parallelism are not supported.
- CPU-native placement is not supported.
- Native routing uses a static even split rather than dynamic load balancing.
- Named models do not participate in streaming reward computation.
- vLLM-Omni reward serving is not implemented.

These capabilities, together with migration of existing reward models to a
unified reward manager, are intentionally left for follow-up RFCs and PRs.
