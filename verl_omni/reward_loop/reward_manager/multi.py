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
"""Modality-neutral reward manager with adapter-based input projection."""

import inspect
import logging
import math
from functools import partial
from typing import Any

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.import_utils import load_extern_object
from verl.utils.reward_score import default_compute_score as _upstream_default_compute_score

from verl_omni.utils.reward_score import default_compute_score_image
from verl_omni.workers.config.reward import get_reward_model_entries, resolve_reward_model_name

from .adapter import (
    AudioRewardAdapter,
    RewardAdapterContext,
    RewardInputAdapter,
    TextRewardAdapter,
    VisualRewardAdapter,
    build_common_reward_kwargs,
)

logger = logging.getLogger(__name__)


def _multi_reward_placeholder(**kwargs: Any) -> None:
    """Sentinel used when configured reward functions own all score calls."""
    raise RuntimeError("_multi_reward_placeholder should never be called directly")


def _filter_kwargs(all_kwargs: dict[str, Any], sig: inspect.Signature) -> dict[str, Any]:
    """Filter kwargs to declared parameters unless the function accepts ``**kwargs``."""
    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return all_kwargs
    return {key: value for key, value in all_kwargs.items() if key in params}


def _is_default_reward_function(compute_score: Any) -> bool:
    return (
        compute_score is None
        or compute_score is _upstream_default_compute_score
        or isinstance(compute_score, partial)
        and compute_score.func is _upstream_default_compute_score
    )


class MultiRewardManager(RewardManagerBase):
    """Score one or many reward terms through modality-specific input adapters.

    The manager owns scorer dispatch, named engine/native executors, failure
    policy, per-term diagnostics, and weighted aggregation. Input adapters own
    only the projection and validation of text, visual, and audio responses.
    Subclasses may override :meth:`_create_reward_adapters` to add a modality
    without duplicating reward execution logic.
    """

    _adapter_mode: str | None = None
    _require_custom_reward_function = False
    _supports_named_reward_models = True

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
    ):
        super().__init__(config, tokenizer, compute_score or _multi_reward_placeholder)
        self.compute_score = compute_score
        if self._adapter_mode in {"visual", "multi_visual"} and (
            compute_score is None or compute_score is _upstream_default_compute_score
        ):
            self.compute_score = default_compute_score_image
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        if self._require_custom_reward_function and _is_default_reward_function(compute_score):
            raise ValueError(f"{type(self).__name__} requires reward.custom_reward_function.")

        self._reward_adapters = self._create_reward_adapters()
        self._engine_reward_executors: dict[str, Any] = {}
        self._native_reward_executors: dict[str, Any] = {}
        self._sub_rewards: list[dict[str, Any]] = []
        self._load_sub_rewards()

    def _create_reward_adapters(self) -> tuple[RewardInputAdapter, ...]:
        """Create the modality hooks used to project rollout outputs."""
        if self._adapter_mode == "visual":
            return (VisualRewardAdapter(required=True),)
        if self._adapter_mode == "audio":
            return (AudioRewardAdapter(required=True),)
        if self._adapter_mode == "multi_visual":
            return (VisualRewardAdapter(required=True), AudioRewardAdapter())
        return (TextRewardAdapter(), VisualRewardAdapter(), AudioRewardAdapter())

    def _load_sub_rewards(self) -> None:
        reward_functions_cfg = self.config.reward.get("reward_functions", {})
        if not reward_functions_cfg:
            return
        reward_models_cfg = get_reward_model_entries(self.config)
        total_weight = 0.0
        reserved_keys = {"path", "name", "weight", "required", "model"}
        for key, entry in reward_functions_cfg.items():
            model_name = resolve_reward_model_name(key, entry, reward_models_cfg)
            path = entry.get("path")
            name = entry.get("name")
            if (path is None) != (name is None):
                raise ValueError(f"Reward function {key!r} must set both path and name")
            if path is None:
                prefix = "Model-backed reward function" if model_name is not None else "Reward function"
                raise ValueError(f"{prefix} {key!r} requires path/name")
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

            extra_args = {field: value for field, value in entry.items() if field not in reserved_keys}
            fn = load_extern_object(path, name)
            self._sub_rewards.append(
                {
                    "key": key,
                    "fn": fn,
                    "weight": weight,
                    "required": required,
                    "sig": inspect.signature(fn),
                    "is_async": inspect.iscoroutinefunction(fn),
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
                inspect.iscoroutinefunction(fn),
            )

        if total_weight <= 0:
            raise ValueError(
                f"Total weight of reward functions must be > 0, got {total_weight}. "
                "Check reward.reward_functions config."
            )

    def _requested_adapter_keys(self) -> frozenset[str]:
        """Return modality kwargs explicitly declared by active scorers."""
        if self._sub_rewards:
            signatures = [sub["sig"] for sub in self._sub_rewards]
        elif not _is_default_reward_function(self.compute_score):
            try:
                signatures = [inspect.signature(self.compute_score)]
            except (TypeError, ValueError):
                signatures = []
        else:
            signatures = []
        return frozenset(name for sig in signatures for name in sig.parameters)

    def set_reward_executors(
        self,
        engine_reward_executors: dict[str, Any] | None,
        native_reward_executors: dict[str, Any] | None,
    ) -> None:
        """Attach per-worker executors for configured engine/native models."""
        self._engine_reward_executors = engine_reward_executors or {}
        self._native_reward_executors = native_reward_executors or {}

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Preserve visual ``(batch, 1)`` and token-aligned score layouts."""
        responses = data.batch["responses"]
        if cls._adapter_mode in {"visual", "multi_visual"} or responses.ndim >= 4:
            return torch.tensor(scores, dtype=torch.float32, device=responses.device).unsqueeze(-1)
        return super().assemble_rm_scores(data, scores)

    async def _build_reward_kwargs(self, data_item: DataProto) -> dict[str, Any]:
        """Project all applicable rollout modalities through adapter hooks."""
        reward_kwargs, extra_info = build_common_reward_kwargs(data_item)
        context = RewardAdapterContext(config=self.config, tokenizer=self.tokenizer, loop=self.loop)
        requested_keys = self._requested_adapter_keys()
        for adapter in self._reward_adapters:
            should_adapt = (
                adapter.required
                or adapter.is_primary(data_item, extra_info)
                or not adapter.provided_keys.isdisjoint(requested_keys)
            )
            if should_adapt and adapter.matches(data_item, extra_info):
                reward_kwargs.update(await adapter.adapt(data_item, extra_info, context))
        return reward_kwargs

    def _legacy_router_kwargs(self, reward_kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.reward_router_address is None:
            return {}
        router_kwargs = {
            "reward_router_address": self.reward_router_address,
            "reward_model_tokenizer": self.reward_model_tokenizer,
            "model_name": self.config.reward.reward_model.model_path,
        }
        if "solution_image" in reward_kwargs:
            rm_rollout = self.config.reward.reward_model.rollout
            sampling_params = {"max_tokens": getattr(rm_rollout, "response_length", None) or 4096}
            if rm_rollout.get("full_determinism", False):
                sampling_params["seed"] = rm_rollout.get("seed", 42)
            router_kwargs["sampling_params"] = sampling_params
        return router_kwargs

    def _resolve_single_compute_score(self, reward_kwargs: dict[str, Any]) -> Any:
        if not _is_default_reward_function(self.compute_score):
            return self.compute_score
        if "solution_audio" in reward_kwargs:
            raise ValueError("Audio rewards require reward.custom_reward_function.")
        if "solution_image" in reward_kwargs:
            return default_compute_score_image
        return self.compute_score or _upstream_default_compute_score

    async def _run_single_reward(self, reward_kwargs: dict[str, Any]) -> dict[str, Any]:
        compute_score = self._resolve_single_compute_score(reward_kwargs)
        call_kwargs = {**reward_kwargs, **self._legacy_router_kwargs(reward_kwargs)}
        try:
            call_kwargs = _filter_kwargs(call_kwargs, inspect.signature(compute_score))
        except (TypeError, ValueError):
            pass
        if inspect.iscoroutinefunction(compute_score):
            result = await compute_score(**call_kwargs)
        else:
            result = await self.loop.run_in_executor(None, lambda: compute_score(**call_kwargs))

        if isinstance(result, dict):
            if "score" not in result:
                raise ValueError("Reward result dictionary is missing 'score'.")
            score = float(result["score"])
            reward_extra_info = {key: value for key, value in result.items() if key != "score"}
        else:
            score = float(result)
            reward_extra_info = {"acc": score}
        if "solution_audio" in reward_kwargs and not math.isfinite(score):
            raise ValueError(f"Audio reward must be finite, got {score!r}.")
        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    async def _run_multi_reward(self, all_kwargs: dict[str, Any]) -> dict[str, Any]:
        combined_score = 0.0
        reward_extra_info: dict[str, Any] = {}
        legacy_router_kwargs = self._legacy_router_kwargs(all_kwargs)

        for sub in self._sub_rewards:
            key = sub["key"]
            fn = sub["fn"]
            weight = sub["weight"]
            required = sub["required"]
            sig = sub["sig"]
            is_async = sub["is_async"]
            model_name = sub["model"]
            sub_kwargs = {**all_kwargs, **legacy_router_kwargs, **sub["extra_args"]}
            filtered_kwargs = _filter_kwargs(sub_kwargs, sig)

            if model_name is not None:
                executor = self._engine_reward_executors.get(model_name)
                if executor is None:
                    executor = self._native_reward_executors.get(model_name)
                if executor is None:
                    raise RuntimeError(f"Reward model {model_name!r} is not available in this worker")
                model_reward_kwargs = getattr(executor, "reward_kwargs", None)
                if model_reward_kwargs is None:
                    raise RuntimeError(f"Reward model {model_name!r} cannot be used with a reward function")
                filtered_kwargs = _filter_kwargs({**sub_kwargs, **model_reward_kwargs()}, sig)

            try:
                if is_async:
                    result = await fn(**filtered_kwargs)
                else:
                    result = await self.loop.run_in_executor(None, lambda f=fn, kw=filtered_kwargs: f(**kw))
                if isinstance(result, dict):
                    score = float(result["score"])
                    reward_extra_info.update(
                        {f"reward/{key}/{field}": value for field, value in result.items() if field != "score"}
                    )
                else:
                    score = float(result)
            except Exception as exc:
                if required:
                    raise RuntimeError(f"Required sub-reward '{key}' failed: {exc}") from exc
                logger.exception(
                    "Sub-reward '%s' raised an exception: %s. Contributing 0 to weighted sum.",
                    key,
                    exc,
                )
                reward_extra_info[f"reward/{key}/errors"] = 1
                score = 0.0

            reward_extra_info[f"reward/{key}"] = score
            combined_score += weight * score

        reward_extra_info["reward/combined"] = combined_score
        return {"reward_score": combined_score, "reward_extra_info": reward_extra_info}

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """Score one rollout sample through the configured adapter and scorer set."""
        if len(data) != 1:
            raise ValueError(f"{type(self).__name__} scores one sample at a time, got batch size {len(data)}.")
        reward_kwargs = await self._build_reward_kwargs(data[0])
        if self._sub_rewards:
            return await self._run_multi_reward(reward_kwargs)
        return await self._run_single_reward(reward_kwargs)


class MultiVisualRewardManager(MultiRewardManager):
    """Backward-compatible visual-only wrapper for :class:`MultiRewardManager`."""

    _adapter_mode = "multi_visual"
