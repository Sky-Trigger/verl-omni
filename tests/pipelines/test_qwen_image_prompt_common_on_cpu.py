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

"""CPU tests for shared Qwen-Image rollout prompt preparation."""

import torch

from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageTokenIdPromptMixin


class _PromptAdapter(QwenImageTokenIdPromptMixin):
    device = torch.device("cpu")

    def __init__(self):
        self.cfg_checks = []
        self.encoded_prompt_ids = []

    def check_cfg_parallel_validity(self, true_cfg_scale, has_neg_prompt):
        self.cfg_checks.append((true_cfg_scale, has_neg_prompt))

    def encode_prompt(
        self,
        prompt_ids,
        attention_mask=None,
        num_images_per_prompt=1,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        max_sequence_length=1024,
    ):
        self.encoded_prompt_ids.append(prompt_ids)
        if prompt_embeds is None:
            prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
            prompt_embeds = prompt_ids.unsqueeze(-1).float()
            if attention_mask is None:
                attention_mask = torch.ones_like(prompt_ids)
            prompt_embeds_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        return QwenImageTokenIdPromptMixin.encode_prompt(
            self,
            prompt_ids=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )


def test_prepare_prompt_context_normalizes_lists_and_enables_true_cfg():
    adapter = _PromptAdapter()

    context = adapter._prepare_prompt_context(
        prompt_ids=[1, 2, 3],
        prompt_mask=torch.tensor([1, 1, 1]),
        negative_prompt_ids=[4, 5, 6],
        negative_prompt_mask=torch.tensor([1, 1, 1]),
        true_cfg_scale=4.0,
        num_images_per_prompt=2,
        max_sequence_length=2,
    )

    assert context.batch_size == 1
    assert context.do_true_cfg is True
    assert context.prompt_embeds.shape == (2, 2, 1)
    assert context.prompt_embeds_mask.shape == (2, 2)
    assert context.negative_prompt_embeds.shape == (2, 2, 1)
    assert context.negative_prompt_embeds_mask.shape == (2, 2)
    assert all(torch.is_tensor(prompt_ids) for prompt_ids in adapter.encoded_prompt_ids)
    assert adapter.cfg_checks == [(4.0, True)]


def test_prepare_prompt_context_keeps_precomputed_batch_and_disables_true_cfg():
    adapter = _PromptAdapter()
    prompt_embeds = torch.randn(2, 3, 4)
    prompt_embeds_mask = torch.ones(2, 3, dtype=torch.long)
    negative_prompt_embeds = torch.randn(2, 3, 4)
    negative_prompt_embeds_mask = torch.ones(2, 3, dtype=torch.long)

    context = adapter._prepare_prompt_context(
        prompt_ids=None,
        prompt_mask=None,
        negative_prompt_ids=None,
        negative_prompt_mask=None,
        true_cfg_scale=1.0,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        num_images_per_prompt=1,
        max_sequence_length=3,
    )

    assert context.batch_size == 2
    assert context.do_true_cfg is False
    assert torch.equal(context.prompt_embeds, prompt_embeds)
    assert torch.equal(context.prompt_embeds_mask, prompt_embeds_mask)
    assert context.negative_prompt_embeds is None
    assert context.negative_prompt_embeds_mask is None
    assert adapter.cfg_checks == [(1.0, True)]
