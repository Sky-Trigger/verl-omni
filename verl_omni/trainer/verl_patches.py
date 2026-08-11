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

from functools import wraps

import torch

from verl.trainer.distillation import losses as distillation_losses
from verl.trainer.ppo import padding_utils
from verl.utils.metric import AggregationType, Metric


def _response_mask_values(response_mask: torch.Tensor) -> torch.Tensor:
    if response_mask.is_nested:
        return response_mask.bool().to_padded_tensor(False)
    return response_mask.bool()


def apply_opd_padding_patch() -> None:
    """Patch verl V1 synthetic padding for OPD teacher fields and empty metrics."""
    original_padding_template = padding_utils.construct_minimal_padding_template
    if not getattr(original_padding_template, "_verl_omni_opd_padding_patch", False):

        @wraps(original_padding_template)
        def patched_padding_template(source_td, source_tag, eos_token_id):
            sample, tag = original_padding_template(source_td, source_tag, eos_token_id)
            sequence_length = sample["input_ids"].size(0)

            teacher_ids = sample.get("teacher_ids")
            if isinstance(teacher_ids, torch.Tensor):
                sample["teacher_ids"] = teacher_ids.new_full(
                    (sequence_length, *teacher_ids.shape[1:]), eos_token_id
                )

            teacher_logprobs = sample.get("teacher_logprobs")
            if isinstance(teacher_logprobs, torch.Tensor):
                sample["teacher_logprobs"] = teacher_logprobs.new_zeros(
                    (sequence_length, *teacher_logprobs.shape[1:])
                )

            return sample, tag

        patched_padding_template._verl_omni_opd_padding_patch = True
        padding_utils.construct_minimal_padding_template = patched_padding_template

    original_loss_range = distillation_losses.compute_distillation_loss_range
    if not getattr(original_loss_range, "_verl_omni_opd_padding_patch", False):

        @wraps(original_loss_range)
        def patched_loss_range(distillation_losses, response_mask):
            valid_losses = distillation_losses[_response_mask_values(response_mask)]
            if valid_losses.numel() == 0:
                return {
                    "distillation/loss_min": Metric(AggregationType.MIN, float("inf")),
                    "distillation/loss_max": Metric(AggregationType.MAX, float("-inf")),
                }
            return original_loss_range(distillation_losses, response_mask)

        patched_loss_range._verl_omni_opd_padding_patch = True
        distillation_losses.compute_distillation_loss_range = patched_loss_range

    original_estimator = distillation_losses.compute_distillation_loss_reverse_kl_estimator
    if not getattr(original_estimator, "_verl_omni_opd_padding_patch", False):

        @wraps(original_estimator)
        def patched_estimator(config, distillation_config, model_output, data):
            losses, metrics = original_estimator(config, distillation_config, model_output, data)
            if not _response_mask_values(data["response_mask"]).any():
                metrics["distillation/abs_loss"] = Metric(
                    AggregationType.MEAN, losses.new_zeros(())
                )
            return losses, metrics

        patched_estimator._verl_omni_opd_padding_patch = True
        distillation_losses.compute_distillation_loss_reverse_kl_estimator = patched_estimator
        for name, loss_fn in distillation_losses.DISTILLATION_LOSS_REGISTRY.items():
            if loss_fn is original_estimator:
                distillation_losses.DISTILLATION_LOSS_REGISTRY[name] = patched_estimator
