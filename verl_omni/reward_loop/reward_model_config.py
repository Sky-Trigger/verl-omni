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
"""Configuration contracts for named reward models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf
from verl.base_config import BaseConfig

__all__ = [
    "RewardModelPlacementConfig",
    "RewardModelSpec",
    "accelerator_workers_enabled",
    "get_reward_model_entries",
    "has_engine_reward_models",
    "has_native_reward_models",
    "has_reward_models",
    "is_engine_backend",
    "parse_reward_model_placement",
    "resolve_reward_model_name",
    "reward_is_enabled",
    "reward_pool_is_separate",
    "reward_role_required",
    "streaming_reward_enabled",
    "to_mapping",
    "validate_reward_model_terms",
]

_ENGINE_BACKENDS = {"engine"}
_NATIVE_BACKENDS = {"native"}


@dataclass
class RewardModelPlacementConfig(BaseConfig):
    """Placement-group bundle indices assigned to one native reward model."""

    devices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("Native reward model placement.devices must be a non-empty list")
        if any(isinstance(device, bool) or not isinstance(device, int) or device < 0 for device in self.devices):
            raise ValueError("Native reward model placement.devices must contain non-negative integers")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("Native reward model placement.devices must not contain duplicates")


@dataclass
class RewardModelSpec(BaseConfig):
    """Static model metadata copied to reward-loop workers."""

    name: str = ""
    backend: str = ""
    model_path: str | None = None
    router_address: str | None = None
    executor_config: dict[str, Any] = field(default_factory=dict)


def to_mapping(value) -> dict[str, Any]:
    """Convert an OmegaConf mapping to a plain dictionary without resolving values."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return OmegaConf.to_container(value, resolve=False)


def get_reward_model_entries(config):
    """Return configured named reward models."""
    return config.reward.get("models", {}) or {}


def has_reward_models(config) -> bool:
    return bool(get_reward_model_entries(config))


def is_engine_backend(backend: str | None) -> bool:
    return backend in _ENGINE_BACKENDS


def has_engine_reward_models(config) -> bool:
    return any(is_engine_backend(model.get("backend")) for model in get_reward_model_entries(config).values())


def has_native_reward_models(config) -> bool:
    return any(model.get("backend") in _NATIVE_BACKENDS for model in get_reward_model_entries(config).values())


def resolve_reward_model_name(term_name: str, term, models) -> str | None:
    """Resolve an explicit model reference or the same-name shorthand."""
    model_name = term.get("model")
    if model_name is None and term_name in models:
        model_name = term_name
    return model_name


def validate_reward_model_terms(config) -> None:
    """Validate named-model references before allocating model resources."""
    models = get_reward_model_entries(config)
    for term_name, term in (config.reward.get("reward_functions", {}) or {}).items():
        model_name = resolve_reward_model_name(term_name, term, models)
        if model_name is None:
            continue
        if model_name not in models:
            raise ValueError(f"Reward term {term_name!r} references unknown model {model_name!r}")
        has_function = term.get("path") is not None and term.get("name") is not None
        if not has_function:
            raise ValueError(
                f"Reward model {model_name!r} needs path/name in reward term {term_name!r} "
                "to turn model output into a score"
            )


def reward_is_enabled(config) -> bool:
    reward_model = config.reward.get("reward_model", {})
    return bool(reward_model.get("enable", False) or has_reward_models(config))


def reward_role_required(config) -> bool:
    """Whether the reward loop needs the trainer-selected parent resource pool."""
    return bool(config.reward.reward_model.get("enable", False) or has_reward_models(config))


def reward_pool_is_separate(config) -> bool:
    return bool(config.reward.reward_model.get("enable_resource_pool", False))


def streaming_reward_enabled(config) -> bool:
    """Whether the current reward path can run inside streaming rollout workers."""
    if not reward_is_enabled(config):
        return True
    if has_engine_reward_models(config) or has_native_reward_models(config):
        return False
    return bool(config.reward.reward_model.get("enable_resource_pool", False))


def accelerator_workers_enabled(config) -> bool:
    """Whether existing custom reward workers use accelerator placement."""
    reward = config.reward
    accelerator_workers = reward.get("accelerator_workers", {}) or {}
    custom_reward = reward.get("custom_reward_function", {}) or {}
    return bool(accelerator_workers.get("enabled", False) or custom_reward.get("use_accelerator", False))


def parse_reward_model_placement(name: str, value) -> RewardModelPlacementConfig:
    """Build and validate one native model placement schema."""
    placement = to_mapping(value)
    if "devices" not in placement:
        raise ValueError(f"Native reward model {name!r} requires placement.devices as native-pool bundle indices")
    try:
        return RewardModelPlacementConfig(devices=placement["devices"])
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Native reward model {name!r} {exc}") from exc
