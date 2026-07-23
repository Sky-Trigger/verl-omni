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

"""Qwen-Image rollout-side adapter for online diffusion DPO."""

import copy
from dataclasses import replace
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline
from vllm_omni.diffusion.models.qwen_image.rope_utils import txt_seq_lens_from_embeds
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import (
    QwenImageTokenIdPromptMixin,
    apply_true_cfg,
    build_img_shapes,
    coalesce_not_none,
)

__all__ = ["QwenImageDPOPipeline"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dpo")
class QwenImageDPOPipeline(QwenImageTokenIdPromptMixin, QwenImagePipeline):
    """Rollout pipeline that returns DPO training tensors with generated images."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Initialize step execution while preserving the DPO output contract."""
        sampling = state.sampling
        prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask = self._extract_prompt_ids(
            [state.prompt] if state.prompt is not None else []
        )

        if prompt_ids is None:
            raise ValueError(
                f"{self.__class__.__name__}.prepare_encode requires either "
                "'prompt_token_ids' or a text 'prompt' on state.prompt."
            )

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling.num_inference_steps or 50
        sigmas = sampling.sigmas or None
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        true_cfg_scale = coalesce_not_none(sampling.true_cfg_scale, 4.0)
        max_sequence_length = sampling.max_sequence_length or 512

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = kwargs.get("attention_kwargs") or {}
        self._current_timestep = None
        self._interrupt = False

        prompt_ctx = self._prepare_prompt_context(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_mask=negative_prompt_mask,
            true_cfg_scale=true_cfg_scale,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        batch_size = prompt_ctx.batch_size
        prompt_embeds = prompt_ctx.prompt_embeds
        prompt_embeds_mask = prompt_ctx.prompt_embeds_mask
        negative_prompt_embeds = prompt_ctx.negative_prompt_embeds
        negative_prompt_embeds_mask = prompt_ctx.negative_prompt_embeds_mask

        num_channels_latents = self.transformer.in_channels // 4
        # Match full-forward random initialisation in model dtype, then cast
        # the exact same values to fp32 for homogeneous live step state.
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            None,
        ).float()
        timesteps, _ = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32).expand(latents.shape[0])
        else:
            guidance = None

        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_embeds_mask
        state.negative_prompt_embeds = negative_prompt_embeds
        state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = prompt_ctx.do_true_cfg
        state.guidance = guidance
        state.img_shapes = build_img_shapes(height, width, batch_size, self.vae_scale_factor)
        state.txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)
        state.negative_txt_seq_lens = txt_seq_lens_from_embeds(negative_prompt_embeds)
        state.sampling.cfg_normalize = True
        state.extra["height"] = height
        state.extra["width"] = width
        return state

    def denoise_step(self, input_batch, **kwargs: Any) -> torch.Tensor | None:
        """Run one DPO denoising pass while keeping request state in FP32."""
        del kwargs
        if self.interrupt:
            return None

        timestep = input_batch.timesteps
        self._current_timestep = timestep
        self.transformer.do_true_cfg = input_batch.do_true_cfg
        model_latents = input_batch.latents.to(self.transformer.img_in.weight.dtype)
        positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
            latents=model_latents,
            timestep=timestep,
            guidance=input_batch.guidance,
            prompt_embeds=input_batch.prompt_embeds,
            prompt_embeds_mask=input_batch.prompt_embeds_mask,
            img_shapes=input_batch.img_shapes,
            txt_seq_lens=input_batch.txt_seq_lens,
            do_true_cfg=input_batch.do_true_cfg,
            negative_prompt_embeds=input_batch.negative_prompt_embeds,
            negative_prompt_embeds_mask=input_batch.negative_prompt_embeds_mask,
            negative_txt_seq_lens=input_batch.negative_txt_seq_lens,
            extra_transformer_kwargs={"attention_kwargs": self.attention_kwargs, "return_dict": False},
        )
        noise_pred = self.predict_noise_maybe_with_cfg(
            input_batch.do_true_cfg,
            input_batch.true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            input_batch.cfg_normalize,
            output_slice,
        )
        return noise_pred.float()

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Advance one DPO step and retain homogeneous FP32 live latents."""
        del kwargs
        if self.interrupt:
            return

        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred.float(),
            state.current_timestep,
            state.latents.float(),
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        ).float()
        state.step_index += 1

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode and restore online DPO's training-output contract."""
        del kwargs
        self._current_timestep = None
        height = state.extra.get("height", state.sampling.height)
        width = state.extra.get("width", state.sampling.width)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        # The non-step DPO path always decodes an image for reward scoring.
        output = self._decode_latents(state.latents, height, width, "pil")

        return replace(
            output,
            custom_output={
                "latents_clean": state.latents.float(),
                "prompt_embeds": state.prompt_embeds,
                "prompt_embeds_mask": state.prompt_embeds_mask,
                "negative_prompt_embeds": state.negative_prompt_embeds,
                "negative_prompt_embeds_mask": state.negative_prompt_embeds_mask,
            },
            to_cpu=True,
        )

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
    ) -> DiffusionOutput:
        del output_type
        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_token_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is None and prompt_embeds is None:
            return DiffusionOutput(output=None, custom_output={})

        prompt_ctx = self._prepare_prompt_context(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_mask=negative_prompt_mask,
            true_cfg_scale=true_cfg_scale,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        batch_size = prompt_ctx.batch_size
        prompt_embeds = prompt_ctx.prompt_embeds
        prompt_embeds_mask = prompt_ctx.prompt_embeds_mask
        negative_prompt_embeds = prompt_ctx.negative_prompt_embeds
        negative_prompt_embeds_mask = prompt_ctx.negative_prompt_embeds_mask
        do_true_cfg = prompt_ctx.do_true_cfg

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        img_shapes = build_img_shapes(height, width, batch_size, self.vae_scale_factor)

        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)
        negative_txt_seq_lens = txt_seq_lens_from_embeds(negative_prompt_embeds)

        self.scheduler.set_begin_index(0)
        for timestep_value in timesteps:
            if self.interrupt:
                continue

            self._current_timestep = timestep_value
            x = latents.to(self.transformer.img_in.weight.dtype)
            timestep = timestep_value.expand(latents.shape[0]).to(device=x.device, dtype=x.dtype)
            self.transformer.do_true_cfg = do_true_cfg
            noise_pred = self.transformer(
                hidden_states=x,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=x,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

            latents = self.scheduler.step(noise_pred.float(), timestep_value, latents, return_dict=False)[0]

        self._current_timestep = None
        latents_clean = latents.float()
        unpacked_latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        unpacked_latents = unpacked_latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(unpacked_latents.device, unpacked_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            unpacked_latents.device, unpacked_latents.dtype
        )
        unpacked_latents = unpacked_latents / latents_std + latents_mean
        image = self.vae.decode(unpacked_latents, return_dict=False)[0][:, :, 0]

        return DiffusionOutput(
            output=image,
            custom_output={
                "latents_clean": latents_clean,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            },
            to_cpu=True,
        )
