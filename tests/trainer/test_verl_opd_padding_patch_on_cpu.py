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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.trainer.distillation import losses as distillation_losses
from verl.trainer.ppo import padding_utils


def _load_patch_module():
    path = Path(__file__).parents[2] / "verl_omni/trainer/verl_patches.py"
    spec = importlib.util.spec_from_file_location("verl_omni_verl_patches_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_opd_padding_patch_resizes_teacher_outputs_and_handles_empty_metrics():
    patch_module = _load_patch_module()
    original_padding_template = padding_utils.construct_minimal_padding_template
    original_loss_range = distillation_losses.compute_distillation_loss_range
    original_estimator = distillation_losses.compute_distillation_loss_reverse_kl_estimator
    original_registry = distillation_losses.DISTILLATION_LOSS_REGISTRY.copy()

    try:
        patch_module.apply_opd_padding_patch()
        patched_padding_template = padding_utils.construct_minimal_padding_template
        patched_estimator = distillation_losses.DISTILLATION_LOSS_REGISTRY["k1"]
        patch_module.apply_opd_padding_patch()

        assert padding_utils.construct_minimal_padding_template is patched_padding_template
        assert distillation_losses.DISTILLATION_LOSS_REGISTRY["k1"] is patched_estimator

        source = {
            "prompts": torch.tensor([1, 2, 3]),
            "responses": torch.tensor([4, 5]),
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.ones(5, dtype=torch.int64),
            "position_ids": torch.arange(5),
            "response_mask": torch.ones(2, dtype=torch.int64),
            "teacher_ids": torch.arange(5, dtype=torch.int32).unsqueeze(-1),
            "teacher_logprobs": torch.randn(5, 1),
        }
        padding, _ = padding_utils.construct_minimal_padding_template(source, {}, eos_token_id=7)

        assert padding["input_ids"].shape == (2,)
        assert padding["teacher_ids"].shape == (2, 1)
        assert padding["teacher_logprobs"].shape == (2, 1)
        assert padding["teacher_ids"].eq(7).all()
        assert padding["teacher_logprobs"].eq(0).all()

        prompts = torch.nested.as_nested_tensor([torch.tensor([7])], layout=torch.jagged)
        responses = torch.nested.as_nested_tensor([torch.tensor([7])], layout=torch.jagged)
        teacher_logprobs = torch.nested.as_nested_tensor([torch.zeros(2, 1)], layout=torch.jagged)
        student_logprobs = torch.nested.as_nested_tensor([torch.zeros(2)], layout=torch.jagged)
        data = TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "response_mask": torch.zeros(1, 1, dtype=torch.bool),
                "teacher_logprobs": teacher_logprobs,
            },
            batch_size=[1],
        )
        config = SimpleNamespace(distillation_loss=SimpleNamespace(loss_mode="k1"))

        losses, metrics = distillation_losses.DISTILLATION_LOSS_REGISTRY["k1"](
            config=None,
            distillation_config=config,
            model_output={"log_probs": student_logprobs},
            data=data,
        )
        range_metrics = distillation_losses.compute_distillation_loss_range(
            distillation_losses=losses,
            response_mask=data["response_mask"],
        )

        assert metrics["distillation/abs_loss"].aggregate() == 0.0
        assert range_metrics["distillation/loss_min"].aggregate() == float("inf")
        assert range_metrics["distillation/loss_max"].aggregate() == float("-inf")
    finally:
        padding_utils.construct_minimal_padding_template = original_padding_template
        distillation_losses.compute_distillation_loss_range = original_loss_range
        distillation_losses.compute_distillation_loss_reverse_kl_estimator = original_estimator
        distillation_losses.DISTILLATION_LOSS_REGISTRY.clear()
        distillation_losses.DISTILLATION_LOSS_REGISTRY.update(original_registry)
