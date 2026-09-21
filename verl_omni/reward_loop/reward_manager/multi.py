# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modality-neutral multi-reward manager with weighted aggregation."""

import asyncio
import inspect
import logging

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.import_utils import load_extern_object

from verl_omni.workers.config.reward import get_reward_model_entries, resolve_reward_model_name

from .audio import AudioRewardManager
from .media import _reward_extra_info
from .visual import VisualRewardManager, _validate_visual_response

logger = logging.getLogger(__name__)


def _multi_reward_placeholder(**kwargs):
    """Sentinel function used as the upstream custom_reward_function placeholder.

    This is never called directly; MultiRewardManager overrides run_single.
    """
    raise RuntimeError("_multi_reward_placeholder should never be called directly")


def _filter_kwargs(all_kwargs: dict, sig: inspect.Signature) -> dict:
    """Filter kwargs to only those declared in the function signature.

    If the function accepts **kwargs, all arguments are passed through.
    """
    params = sig.parameters
    # Check if the function accepts **kwargs
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return all_kwargs
    # Only pass declared parameters
    return {k: v for k, v in all_kwargs.items() if k in params}


class MultiRewardManager(RewardManagerBase):
    """Load and aggregate reward functions without assuming one response modality.

    Each sub-reward function is called with filtered kwargs (based on its signature),
    and the final reward is a weighted sum of all sub-rewards.

    A sub-reward may reference a named model. The selected executor supplies
    inference access, while the configured reward function owns score semantics.
    """

    _require_visual_response = False

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, _multi_reward_placeholder)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        self._engine_reward_executors = {}
        self._native_reward_executors = {}

        reward_functions_cfg = config.reward.reward_functions
        reward_models_cfg = get_reward_model_entries(config)
        if not reward_functions_cfg:
            raise ValueError("MultiRewardManager requires non-empty reward.reward_functions config")

        self._sub_rewards = []
        total_weight = 0.0
        _reserved_keys = {"path", "name", "weight", "required", "model"}
        for key, entry in reward_functions_cfg.items():
            model_name = resolve_reward_model_name(key, entry, reward_models_cfg)
            path = entry.get("path")
            name = entry.get("name")
            if (path is None) != (name is None):
                raise ValueError(f"Reward function {key!r} must set both path and name")
            if model_name is None and path is None:
                raise ValueError(f"Reward function {key!r} requires path/name")
            if model_name is not None and path is None:
                raise ValueError(f"Model-backed reward function {key!r} requires path/name")
            weight = float(entry.get("weight", 1.0))
            required_value = entry.get("required", False)
            if isinstance(required_value, str):
                normalized = required_value.lower()
                if normalized not in {"true", "false"}:
                    raise ValueError(f"Invalid required value: {required_value!r}")
                required = normalized == "true"
            elif isinstance(required_value, bool):
                required = required_value
            else:
                raise TypeError(f"required must be a boolean, got {type(required_value).__name__}")
            total_weight += weight

            # Collect non-manager fields to pass to compute_score.
            extra_args = {k: v for k, v in entry.items() if k not in _reserved_keys}

            fn = load_extern_object(path, name) if path is not None else None
            sig = inspect.signature(fn) if fn is not None else None
            is_async = inspect.iscoroutinefunction(fn) if fn is not None else True

            self._sub_rewards.append(
                {
                    "key": key,
                    "fn": fn,
                    "weight": weight,
                    "required": required,
                    "sig": sig,
                    "is_async": is_async,
                    "extra_args": extra_args,
                    "model": model_name,
                }
            )
            logger.info(
                "Loaded sub-reward '%s': %s (weight=%s, required=%s, async=%s)",
                key,
                model_name or f"{path}:{name}",
                weight,
                required,
                is_async,
            )

        if total_weight <= 0:
            raise ValueError(
                f"Total weight of reward functions must be > 0, got {total_weight}. "
                f"Check reward.reward_functions config."
            )

    def set_reward_executors(self, engine_reward_executors: dict | None, native_reward_executors: dict | None) -> None:
        """Attach per-worker executors for configured engine/native models."""
        self._engine_reward_executors = engine_reward_executors or {}
        self._native_reward_executors = native_reward_executors or {}

    @staticmethod
    def _is_visual_response(response) -> bool:
        """Return whether one sample carries an image, video, or visual latent."""
        return isinstance(response, torch.Tensor) and response.ndim >= 3

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Preserve visual ``(batch, 1)`` and token-aligned score layouts."""
        responses = data.batch["responses"]
        if cls._require_visual_response or responses.ndim >= 4:
            return torch.tensor(scores, dtype=torch.float32, device=responses.device).unsqueeze(-1)
        return super().assemble_rm_scores(data, scores)

    async def _build_reward_kwargs(self, data_item: DataProto) -> dict:
        """Project available text and media outputs into scorer keyword arguments."""
        batch = data_item.non_tensor_batch
        response = data_item.batch["responses"]
        extra_info = _reward_extra_info(data_item)
        extra_info["num_turns"] = batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = batch.get("reward_scores", {})
        if "global_steps" in batch:
            extra_info["global_steps"] = batch["global_steps"]

        reward_kwargs = {
            "data_source": batch["data_source"],
            "ground_truth": batch["reward_model"]["ground_truth"],
            "extra_info": extra_info,
        }

        media_kind = extra_info.get("media_kind")
        if media_kind is not None and media_kind not in {"image", "video", "audio"}:
            raise ValueError(f"Unsupported reward media kind: {media_kind!r}")

        if self._require_visual_response or media_kind in {"image", "video"} or self._is_visual_response(response):
            _validate_visual_response(response, self.config, is_validate=data_item.meta_info.get("validate", False))
            reward_kwargs["solution_image"] = response

        if media_kind == "audio" or "audio" in extra_info or "audio_sample_rate" in extra_info:
            reward_kwargs["solution_audio"] = await asyncio.to_thread(AudioRewardManager._extract_audio, extra_info)

        if isinstance(response, torch.Tensor) and response.ndim == 1 and "attention_mask" in data_item.batch:
            response_length = response.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
            valid_response_ids = response[:valid_response_length]
            reward_kwargs["solution_str"] = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            )

        if self.reward_router_address is not None:
            reward_kwargs.update(
                reward_router_address=self.reward_router_address,
                reward_model_tokenizer=self.reward_model_tokenizer,
                model_name=self.config.reward.reward_model.model_path,
            )
        return reward_kwargs

    async def run_single(self, data: DataProto) -> dict:
        if len(data) != 1:
            raise ValueError(f"MultiRewardManager scores one sample at a time, got batch size {len(data)}.")
        data_item = data[0]
        all_kwargs = await self._build_reward_kwargs(data_item)

        combined_score = 0.0
        reward_extra_info = {}

        for sub in self._sub_rewards:
            key = sub["key"]
            fn = sub["fn"]
            weight = sub["weight"]
            required = sub["required"]
            sig = sub["sig"]
            is_async = sub["is_async"]
            extra_args = sub["extra_args"]
            model_name = sub["model"]

            # Merge per-reward extra config fields into kwargs
            sub_kwargs = {**all_kwargs, **extra_args}
            filtered_kwargs = _filter_kwargs(sub_kwargs, sig) if sig is not None else {}

            if model_name is not None:
                executor = self._engine_reward_executors.get(model_name)
                if executor is None:
                    executor = self._native_reward_executors.get(model_name)
                if executor is None:
                    raise RuntimeError(f"Reward model {model_name!r} is not available in this worker")
                reward_kwargs = getattr(executor, "reward_kwargs", None)
                if reward_kwargs is None:
                    raise RuntimeError(f"Reward model {model_name!r} cannot be used with a reward function")
                filtered_kwargs = _filter_kwargs({**sub_kwargs, **reward_kwargs()}, sig)

            try:
                if is_async:
                    result = await fn(**filtered_kwargs)
                else:
                    result = await self.loop.run_in_executor(None, lambda f=fn, kw=filtered_kwargs: f(**kw))

                if isinstance(result, dict):
                    score = float(result["score"])
                    for rk, rv in result.items():
                        if rk == "score":
                            continue
                        reward_extra_info[f"reward/{key}/{rk}"] = rv
                else:
                    score = float(result)
            except Exception as e:
                if required:
                    raise RuntimeError(f"Required sub-reward '{key}' failed: {e}") from e
                logger.exception(
                    "Sub-reward '%s' raised an exception: %s. Contributing 0 to weighted sum.",
                    key,
                    e,
                )
                reward_extra_info[f"reward/{key}/errors"] = 1
                score = 0.0

            reward_extra_info[f"reward/{key}"] = score
            combined_score += weight * score

        reward_extra_info["reward/combined"] = combined_score
        return {"reward_score": combined_score, "reward_extra_info": reward_extra_info}


class MultiVisualRewardManager(MultiRewardManager, VisualRewardManager):
    """Backward-compatible visual-only alias for :class:`MultiRewardManager`."""

    _require_visual_response = True
