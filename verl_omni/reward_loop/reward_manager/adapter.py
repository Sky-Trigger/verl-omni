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
"""Input adapters shared by single- and multi-reward managers."""

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from verl import DataProto

from .media import _reward_extra_info

SUPPORTED_MEDIA_KINDS = frozenset({"image", "video", "audio"})


@dataclass(frozen=True)
class RewardAdapterContext:
    """Runtime dependencies made available to reward input adapters."""

    config: Any
    tokenizer: Any
    loop: asyncio.AbstractEventLoop


def build_common_reward_kwargs(data_item: DataProto) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build modality-independent scorer arguments for one rollout sample."""
    batch = data_item.non_tensor_batch
    extra_info = _reward_extra_info(data_item)
    extra_info["num_turns"] = batch.get("__num_turns__", None)
    extra_info["rollout_reward_scores"] = batch.get("reward_scores", {})
    if "global_steps" in batch:
        extra_info["global_steps"] = batch["global_steps"]
    media_kind = extra_info.get("media_kind")
    if media_kind is not None and media_kind not in SUPPORTED_MEDIA_KINDS:
        raise ValueError(f"Unsupported reward media kind: {media_kind!r}")
    return (
        {
            "data_source": batch["data_source"],
            "ground_truth": batch["reward_model"]["ground_truth"],
            "extra_info": extra_info,
        },
        extra_info,
    )


class RewardInputAdapter:
    """Hook that projects one rollout modality into scorer keyword arguments."""

    def matches(self, data_item: DataProto, extra_info: dict[str, Any]) -> bool:
        """Return whether this adapter applies to the sample."""
        raise NotImplementedError

    async def adapt(
        self,
        data_item: DataProto,
        extra_info: dict[str, Any],
        context: RewardAdapterContext,
    ) -> dict[str, Any]:
        """Return scorer keyword arguments contributed by this adapter."""
        raise NotImplementedError


class TextRewardAdapter(RewardInputAdapter):
    """Decode token responses for text reward functions."""

    def matches(self, data_item: DataProto, extra_info: dict[str, Any]) -> bool:
        response = data_item.batch["responses"]
        return isinstance(response, torch.Tensor) and response.ndim == 1 and "attention_mask" in data_item.batch

    async def adapt(
        self,
        data_item: DataProto,
        extra_info: dict[str, Any],
        context: RewardAdapterContext,
    ) -> dict[str, Any]:
        response = data_item.batch["responses"]
        response_length = response.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response[:valid_response_length]
        solution_str = await context.loop.run_in_executor(
            None,
            lambda: context.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )
        return {"solution_str": solution_str}


def validate_visual_response(response_visual: Any, config: Any, *, is_validate: bool) -> None:
    """Validate a generated image, video, or visual latent tensor."""
    rollout_config = config.actor_rollout_ref.rollout
    pipeline_config = rollout_config.val_kwargs.pipeline if is_validate else rollout_config.pipeline
    output_type = pipeline_config.get("output_type", "image")

    if output_type == "latent":
        if not isinstance(response_visual, torch.Tensor) or not response_visual.dtype.is_floating_point:
            dtype = getattr(response_visual, "dtype", type(response_visual))
            raise ValueError(f"Expected floating-point latent responses, got {dtype}.")
    elif not isinstance(response_visual, torch.Tensor) or response_visual.dtype != torch.uint8:
        dtype = getattr(response_visual, "dtype", type(response_visual))
        raise ValueError(f"Expected uint8 pixel responses for output_type={output_type!r}, got {dtype}.")


class VisualRewardAdapter(RewardInputAdapter):
    """Project and validate image, video, and visual-latent responses."""

    def __init__(self, *, required: bool = False):
        self.required = required

    def matches(self, data_item: DataProto, extra_info: dict[str, Any]) -> bool:
        response = data_item.batch["responses"]
        return (
            self.required
            or extra_info.get("media_kind") in {"image", "video"}
            or isinstance(response, torch.Tensor)
            and response.ndim >= 3
        )

    async def adapt(
        self,
        data_item: DataProto,
        extra_info: dict[str, Any],
        context: RewardAdapterContext,
    ) -> dict[str, Any]:
        response = data_item.batch["responses"]
        validate_visual_response(response, context.config, is_validate=data_item.meta_info.get("validate", False))
        return {"solution_image": response}


def extract_audio(extra_info: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Validate and normalize one generated mono waveform."""
    audio = extra_info.get("audio")
    sample_rate = extra_info.get("audio_sample_rate")
    if audio is None:
        raise KeyError("Audio reward requires extra_info['audio'] from the rollout.")
    if sample_rate is None:
        raise KeyError("Audio reward requires extra_info['audio_sample_rate'] from the rollout.")

    try:
        if isinstance(audio, np.ndarray) and audio.dtype == object:
            audio = np.asarray(audio, dtype=np.float32)
        waveform = torch.as_tensor(audio).detach().float().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Audio reward could not convert the waveform to numeric samples.") from exc
    while waveform.ndim > 1 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim != 1:
        raise ValueError(
            f"Expected one mono waveform with shape (T,) or leading singleton dimensions, got {tuple(waveform.shape)}."
        )
    if waveform.numel() == 0:
        raise ValueError("Audio reward received an empty waveform.")
    if not torch.isfinite(waveform).all():
        raise ValueError("Audio reward received a waveform containing NaN or infinity.")

    if isinstance(sample_rate, np.ndarray | torch.Tensor):
        sample_rate_count = sample_rate.size if isinstance(sample_rate, np.ndarray) else sample_rate.numel()
        if sample_rate_count != 1:
            raise ValueError("Audio reward requires one scalar sample rate per waveform.")
        sample_rate = sample_rate.item()
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int | float):
        raise TypeError(f"Audio sample rate must be numeric, got {type(sample_rate).__name__}.")
    if not math.isfinite(float(sample_rate)) or float(sample_rate) <= 0 or float(sample_rate) != int(sample_rate):
        raise ValueError(f"Audio sample rate must be a positive integer, got {sample_rate!r}.")
    return waveform.numpy().astype(np.float32, copy=False), int(sample_rate)


class AudioRewardAdapter(RewardInputAdapter):
    """Project and validate a generated audio waveform."""

    def __init__(
        self,
        *,
        required: bool = False,
        audio_extractor: Callable[[dict[str, Any]], tuple[np.ndarray, int]] = extract_audio,
    ):
        self.required = required
        self.audio_extractor = audio_extractor

    def matches(self, data_item: DataProto, extra_info: dict[str, Any]) -> bool:
        return (
            self.required
            or extra_info.get("media_kind") == "audio"
            or "audio" in extra_info
            or "audio_sample_rate" in extra_info
        )

    async def adapt(
        self,
        data_item: DataProto,
        extra_info: dict[str, Any],
        context: RewardAdapterContext,
    ) -> dict[str, Any]:
        solution_audio = await asyncio.to_thread(self.audio_extractor, extra_info)
        return {"solution_audio": solution_audio}
