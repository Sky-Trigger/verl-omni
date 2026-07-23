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

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QwenImagePromptContext:
    """Prompt tensors shared by Qwen-Image rollout adapters."""

    batch_size: int
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    do_true_cfg: bool


QWEN_IMAGE_VAE_SCALE_FACTOR = 8


def coalesce_not_none(value, default):
    return default if value is None else value


def build_img_shapes(
    height: int, width: int, batch_size: int, vae_scale_factor: int
) -> list[list[tuple[int, int, int]]]:
    latent_height = height // vae_scale_factor // 2
    latent_width = width // vae_scale_factor // 2
    return [[(1, latent_height, latent_width)]] * batch_size


def apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    comb_pred = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)


class QwenImageTokenIdPromptMixin:
    """Encode pre-tokenized Qwen-Image prompts for rollout adapters."""

    def _extract_prompt_ids(self, prompts):
        """Extract tokenized prompts, with a raw-text warm-up fallback."""
        prompt_ids = None
        prompt_mask = None
        negative_prompt_ids = None
        negative_prompt_mask = None
        if prompts:
            prompt = prompts[0]
            if isinstance(prompt, dict):
                prompt_ids = prompt.get("prompt_token_ids")
                prompt_mask = prompt.get("prompt_mask")
                negative_prompt_ids = prompt.get("negative_prompt_ids")
                negative_prompt_mask = prompt.get("negative_prompt_mask")
                if prompt_ids is None and prompt.get("prompt"):
                    prompt_ids, prompt_mask = self._tokenize_text_prompt(prompt["prompt"])
                if negative_prompt_ids is None and prompt.get("negative_prompt"):
                    negative_prompt_ids, negative_prompt_mask = self._tokenize_text_prompt(prompt["negative_prompt"])
            elif isinstance(prompt, str):
                prompt_ids, prompt_mask = self._tokenize_text_prompt(prompt)
        return prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask

    def _tokenize_text_prompt(self, text: str | list[str]):
        """Tokenize raw text with the Qwen chat template."""
        prompt = [text] if isinstance(text, str) else text
        formatted = [self.prompt_template_encode.format(item) for item in prompt]
        tokens = self.tokenizer(
            formatted,
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return tokens.input_ids, tokens.attention_mask

    def _get_qwen_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        drop_idx = self.prompt_template_encode_start_idx
        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ):
        if prompt_embeds is None:
            if prompt_ids is None:
                raise ValueError("`prompt_ids` must be provided when `prompt_embeds` is None.")
            prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
            attention_mask = (
                attention_mask.unsqueeze(0)
                if attention_mask is not None and attention_mask.ndim == 1
                else attention_mask
            )
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt_ids, attention_mask=attention_mask)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def _prepare_prompt_context(
        self,
        *,
        prompt_ids: torch.Tensor | list[int] | None,
        prompt_mask: torch.Tensor | None,
        negative_prompt_ids: torch.Tensor | list[int] | None,
        negative_prompt_mask: torch.Tensor | None,
        true_cfg_scale: float,
        num_images_per_prompt: int,
        max_sequence_length: int,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
    ) -> QwenImagePromptContext:
        """Prepare text conditioning without algorithm-specific latent state.

        Qwen-Image-Edit overrides ``encode_prompt`` because it also consumes
        condition images and therefore intentionally does not use this helper.
        """
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        if prompt_ids is not None:
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Qwen-Image rollout requires either `prompt_ids` or `prompt_embeds`.")

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        return QwenImagePromptContext(
            batch_size=batch_size,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            do_true_cfg=do_true_cfg,
        )
