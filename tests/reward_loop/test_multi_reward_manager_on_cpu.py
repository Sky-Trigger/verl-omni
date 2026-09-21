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
"""CPU tests for modality-neutral multi-reward aggregation."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.reward import resolve_reward_manager_cls

from verl_omni.reward_loop.reward_manager.adapter import AudioRewardAdapter, TextRewardAdapter, VisualRewardAdapter
from verl_omni.reward_loop.reward_manager.multi import MultiRewardManager, MultiVisualRewardManager, _filter_kwargs
from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager

# Path to this file — load_extern_object will import dummy functions from here.
DUMMY_REWARDS_PATH = "tests/reward_loop/test_multi_reward_manager_on_cpu.py"


# ---------------------------------------------------------------------------
# Dummy reward functions (loaded by MultiRewardManager via load_extern_object)
# ---------------------------------------------------------------------------


def reward_fixed_score(data_source, solution_image, ground_truth, extra_info):
    """Always returns 0.5."""
    return 0.5


def reward_dict_result(data_source, ground_truth):
    """Returns a dict with score and extra metadata."""
    return {"score": 1.0, "detail": "perfect"}


def reward_raises(data_source, solution_image, ground_truth, extra_info):
    """Always raises to test error handling."""
    raise ValueError("intentional failure")


async def reward_async(data_source, solution_image, ground_truth, extra_info):
    """Async reward that returns 0.8."""
    return 0.8


async def reward_asserts_uint8_contract(data_source, solution_image, ground_truth, extra_info):
    """Verify reward managers preserve the uint8 response contract."""
    assert solution_image.dtype == torch.uint8
    return int(solution_image[0, 0, 0])


async def reward_asserts_float_latent_contract(data_source, solution_image, ground_truth, extra_info):
    """Verify reward managers preserve floating-point latent responses."""
    assert solution_image.dtype == torch.float32
    assert solution_image.shape == (16, 2, 2)
    return float(solution_image[0, 0, 0])


def reward_text(solution_str, ground_truth):
    assert solution_str == "decoded response"
    assert ground_truth == "hello"
    return {"score": 0.4, "modality": "text"}


def reward_audio(solution_audio, extra_info):
    waveform, sample_rate = solution_audio
    assert waveform.dtype == np.float32
    np.testing.assert_array_equal(waveform, np.arange(8, dtype=np.float32))
    assert sample_rate == 24_000
    assert extra_info["media_kind"] == "audio"
    return {"score": 0.7, "modality": "audio"}


def reward_visual_with_aux_audio(solution_image, solution_audio):
    waveform, sample_rate = solution_audio
    assert solution_image.dtype == torch.uint8
    np.testing.assert_array_equal(waveform, np.arange(8, dtype=np.float32))
    assert sample_rate == 32_000
    return 0.9


async def reward_uses_named_engine_router(reward_router_address, model_name):
    assert reward_router_address == "engine-router"
    assert model_name == "ocr-model"
    return {"score": 0.6, "backend": "engine-function"}


async def reward_uses_native_model(reward_model, ground_truth, solution_image):
    output = await reward_model.infer(prompt=ground_truth, image=solution_image)
    return {"score": output["value"], "backend": "native-function"}


class _NativeModelExecutor:
    def reward_kwargs(self):
        return {"reward_model": self}

    async def infer(self, prompt, image):
        assert prompt == "hello"
        assert image.dtype == torch.uint8
        return {"value": 0.75}


class _EngineRouterClient:
    def reward_kwargs(self):
        return {"reward_router_address": "engine-router", "model_name": "ocr-model"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(reward_functions: dict):
    """Build a minimal config with reward_functions populated."""
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")

    config.reward.reward_functions = OmegaConf.create(reward_functions)
    config.reward.reward_model.enable = False
    return config


def _make_single_data(data_source: str = "test_source") -> DataProto:
    """Create a single-item DataProto for run_single."""
    return DataProto.from_dict(
        tensors={"responses": torch.randint(256, (1, 3, 64, 64), dtype=torch.uint8)},
        non_tensors={
            "data_source": [data_source],
            "reward_model": [{"ground_truth": "hello"}],
            "extra_info": [{}],
        },
    )


def _build_manager(reward_functions: dict) -> MultiRewardManager:
    config = _make_config(reward_functions)
    tokenizer = MagicMock()
    return MultiRewardManager(config, tokenizer, compute_score=None)


def _build_visual_latent_manager() -> VisualRewardManager:
    config = _make_config({})
    OmegaConf.update(config, "actor_rollout_ref.rollout.pipeline.output_type", "latent", force_add=True)
    return VisualRewardManager(config, MagicMock(), reward_asserts_float_latent_contract)


def _build_latent_multi_manager() -> MultiRewardManager:
    manager = _build_manager(
        {
            "latent": {
                "path": DUMMY_REWARDS_PATH,
                "name": "reward_asserts_float_latent_contract",
                "weight": 1.0,
            }
        }
    )
    OmegaConf.update(manager.config, "actor_rollout_ref.rollout.pipeline.output_type", "latent", force_add=True)
    return manager


def _build_visual_pixel_manager() -> VisualRewardManager:
    return VisualRewardManager(_make_config({}), MagicMock(), reward_asserts_uint8_contract)


def _build_pixel_multi_manager() -> MultiRewardManager:
    return _build_manager(
        {
            "pixel": {
                "path": DUMMY_REWARDS_PATH,
                "name": "reward_asserts_uint8_contract",
                "weight": 1.0,
            }
        }
    )


# ---------------------------------------------------------------------------
# _filter_kwargs
# ---------------------------------------------------------------------------


class TestFilterKwargs:
    def test_filters_to_declared_params(self):
        import inspect

        def fn(a, b):
            pass

        sig = inspect.signature(fn)
        result = _filter_kwargs({"a": 1, "b": 2, "c": 3}, sig)
        assert result == {"a": 1, "b": 2}

    def test_passes_all_when_var_keyword(self):
        import inspect

        def fn(a, **kwargs):
            pass

        sig = inspect.signature(fn)
        result = _filter_kwargs({"a": 1, "b": 2, "c": 3}, sig)
        assert result == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------------------
# MultiRewardManager.run_single
# ---------------------------------------------------------------------------


class TestVisualRewardManagerDefaults:
    def test_default_jpeg_reward_accepts_manager_call_contract(self):
        manager = VisualRewardManager(_make_config({}), MagicMock(), compute_score=None)
        data = _make_single_data(data_source="jpeg_compressibility")
        data.batch["responses"] = torch.zeros_like(data.batch["responses"])

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] < 0
        assert result["reward_extra_info"]["acc"] == pytest.approx(result["reward_score"])


class TestMultiRewardManagerRunSingle:
    @pytest.mark.parametrize("manager_factory", [_build_visual_latent_manager, _build_latent_multi_manager])
    def test_float_latent_response_is_forwarded_to_reward(self, manager_factory):
        manager = manager_factory()
        data = _make_single_data()
        data.batch["responses"] = torch.full((1, 16, 2, 2), 0.5)

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.5)

    @pytest.mark.parametrize("manager_factory", [_build_visual_pixel_manager, _build_pixel_multi_manager])
    def test_float_pixel_response_is_rejected(self, manager_factory):
        manager = manager_factory()
        data = _make_single_data()
        data.batch["responses"] = data.batch["responses"].float()

        with pytest.raises(
            ValueError,
            match=r"Expected uint8 pixel responses for output_type='image', got torch\.float32\.",
        ):
            manager.loop.run_until_complete(manager.run_single(data))

    @pytest.mark.parametrize("manager_factory", [_build_visual_latent_manager, _build_latent_multi_manager])
    def test_uint8_latent_response_is_rejected(self, manager_factory):
        manager = manager_factory()
        data = _make_single_data()

        with pytest.raises(ValueError, match=r"Expected floating-point latent responses, got torch\.uint8\."):
            manager.loop.run_until_complete(manager.run_single(data))

    @pytest.mark.parametrize("manager_factory", [_build_visual_pixel_manager, _build_pixel_multi_manager])
    def test_validation_uses_validation_output_type(self, manager_factory):
        manager = manager_factory()
        OmegaConf.update(
            manager.config,
            "actor_rollout_ref.rollout.val_kwargs.pipeline.output_type",
            "latent",
            force_add=True,
        )
        manager.compute_score = reward_asserts_float_latent_contract
        manager.is_async_reward_score = True
        if manager._sub_rewards:
            manager._sub_rewards[0]["fn"] = reward_asserts_float_latent_contract
            manager._sub_rewards[0]["is_async"] = True

        data = _make_single_data()
        data.batch["responses"] = torch.full((1, 16, 2, 2), 0.5)
        data.meta_info["validate"] = True

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.5)

    def test_weighted_aggregation(self):
        """Two reward functions with different weights produce correct combined score."""
        reward_fns = {
            "fixed": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 2.0},
            "dict_result": {"path": DUMMY_REWARDS_PATH, "name": "reward_dict_result", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        result = manager.loop.run_until_complete(manager.run_single(data))

        # combined = 2.0 * 0.5 + 1.0 * 1.0 = 2.0
        assert result["reward_score"] == pytest.approx(2.0)
        assert result["reward_extra_info"]["reward/fixed"] == pytest.approx(0.5)
        assert result["reward_extra_info"]["reward/dict_result"] == pytest.approx(1.0)
        assert result["reward_extra_info"]["reward/dict_result/detail"] == "perfect"
        assert "reward/dict_result/score" not in result["reward_extra_info"]
        assert result["reward_extra_info"]["reward/combined"] == pytest.approx(2.0)

    def test_required_exception_fails_fast(self):
        """A failing required sub-reward aborts reward computation."""
        reward_fns = {
            "good": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0},
            "bad": {
                "path": DUMMY_REWARDS_PATH,
                "name": "reward_raises",
                "weight": 1.0,
                "required": True,
            },
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        with pytest.raises(RuntimeError, match="Required sub-reward 'bad' failed: intentional failure"):
            manager.loop.run_until_complete(manager.run_single(data))

    def test_optional_exception_contributes_zero(self):
        """A failing optional sub-reward records the error and contributes zero."""
        reward_fns = {
            "good": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 2.0},
            "bad": {"path": DUMMY_REWARDS_PATH, "name": "reward_raises", "weight": 3.0},
        }
        manager = _build_manager(reward_fns)

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(1.0)
        assert result["reward_extra_info"]["reward/bad"] == pytest.approx(0.0)
        assert result["reward_extra_info"]["reward/bad/errors"] == 1

    def test_async_reward_function(self):
        """Async reward functions are awaited correctly."""
        reward_fns = {
            "async_fn": {"path": DUMMY_REWARDS_PATH, "name": "reward_async", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.8)

    def test_mixes_rule_engine_and_native_models(self):
        manager = _build_manager(
            {
                "rule": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0},
                "engine": {
                    "model": "ocr_engine",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_named_engine_router",
                    "weight": 2.0,
                },
                "native": {
                    "model": "native_pickscore",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_native_model",
                    "weight": 1.0,
                },
            }
        )
        manager.set_reward_executors(
            {"ocr_engine": _EngineRouterClient()}, {"native_pickscore": _NativeModelExecutor()}
        )

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(2.45)
        assert result["reward_extra_info"]["reward/rule"] == pytest.approx(0.5)
        assert result["reward_extra_info"]["reward/engine"] == pytest.approx(0.6)
        assert result["reward_extra_info"]["reward/native"] == pytest.approx(0.75)
        assert result["reward_extra_info"]["reward/engine/backend"] == "engine-function"
        assert result["reward_extra_info"]["reward/native/backend"] == "native-function"

    def test_native_reward_function_receives_only_inference_handle(self):
        manager = _build_manager(
            {
                "native": {
                    "model": "native_model",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_native_model",
                }
            }
        )
        manager.set_reward_executors(None, {"native_model": _NativeModelExecutor()})

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.75)
        assert result["reward_extra_info"]["reward/native/backend"] == "native-function"

    def test_engine_reward_function_uses_its_named_router(self):
        manager = _build_manager(
            {
                "ocr": {
                    "model": "ocr_engine",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_named_engine_router",
                    "weight": 1.0,
                },
            }
        )
        manager.set_reward_executors({"ocr_engine": _EngineRouterClient()}, None)

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.6)
        assert result["reward_extra_info"]["reward/ocr/backend"] == "engine-function"

    def test_jpeg_reward_via_file_path(self):
        """JPEG reward loaded via file path must not fail on relative imports."""
        reward_fns = {
            "jpeg": {
                "path": "verl_omni/utils/reward_score/jpeg_compressibility.py",
                "name": "compute_score",
                "weight": 1.0,
            },
        }
        manager = _build_manager(reward_fns)
        data = DataProto.from_dict(
            tensors={"responses": torch.randint(256, (1, 3, 64, 64), dtype=torch.uint8)},
            non_tensors={
                "data_source": ["jpeg_compressibility"],
                "reward_model": [{"ground_truth": "hello"}],
                "extra_info": [{}],
            },
        )

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] != pytest.approx(0.0)
        assert "reward/jpeg" in result["reward_extra_info"]

    def test_uint8_response_is_forwarded_to_custom_reward(self):
        reward_fns = {
            "contract": {"path": DUMMY_REWARDS_PATH, "name": "reward_asserts_uint8_contract", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()
        data.batch["responses"] = torch.full_like(data.batch["responses"], 128, dtype=torch.uint8)

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(128)

    def test_text_reward_receives_decoded_response(self):
        manager = _build_manager({"text": {"path": DUMMY_REWARDS_PATH, "name": "reward_text", "weight": 2.0}})
        manager.tokenizer.decode.return_value = "decoded response"
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[1, 2]]),
                "responses": torch.tensor([[3, 4, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            },
            non_tensors={
                "data_source": ["text"],
                "reward_model": [{"ground_truth": "hello"}],
                "extra_info": [{}],
            },
        )

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.8)
        assert result["reward_extra_info"]["reward/text/modality"] == "text"
        manager.tokenizer.decode.assert_called_once()

    def test_audio_reward_receives_validated_waveform(self):
        manager = _build_manager({"audio": {"path": DUMMY_REWARDS_PATH, "name": "reward_audio", "weight": 1.0}})
        data = DataProto.from_dict(
            tensors={"responses": torch.tensor([[1, 2]])},
            non_tensors={
                "data_source": ["audio"],
                "reward_model": [{"ground_truth": "hello"}],
                "extra_info": [{}],
                "tool_extra_fields": [
                    {
                        "audio": np.arange(8, dtype=np.float32),
                        "audio_sample_rate": 24_000,
                        "media_kind": "audio",
                    }
                ],
            },
        )

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.7)
        assert result["reward_extra_info"]["reward/audio/modality"] == "audio"

    def test_single_audio_reward_uses_generic_manager_adapter(self):
        manager = MultiRewardManager(_make_config({}), MagicMock(), compute_score=reward_audio)
        data = DataProto.from_dict(
            tensors={"responses": torch.tensor([[1, 2]])},
            non_tensors={
                "data_source": ["audio"],
                "reward_model": [{"ground_truth": "hello"}],
                "extra_info": [{}],
                "tool_extra_fields": [
                    {
                        "audio": np.arange(8, dtype=np.float32),
                        "audio_sample_rate": 24_000,
                        "media_kind": "audio",
                    }
                ],
            },
        )

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result == {"reward_score": pytest.approx(0.7), "reward_extra_info": {"modality": "audio"}}

    def test_single_visual_reward_preserves_legacy_engine_router_contract(self):
        async def score(reward_router_address, model_name, sampling_params, solution_image):
            assert reward_router_address == "legacy-router"
            assert model_name == "legacy-model"
            assert sampling_params == {"max_tokens": 2048, "seed": 17}
            assert solution_image.dtype == torch.uint8
            return 0.6

        config = _make_config({})
        config.reward.reward_model.model_path = "legacy-model"
        config.reward.reward_model.rollout.full_determinism = True
        config.reward.reward_model.rollout.seed = 17
        manager = MultiRewardManager(
            config,
            MagicMock(),
            compute_score=score,
            reward_router_address="legacy-router",
        )

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.6)

    def test_declared_audio_without_waveform_fails_before_scoring(self):
        manager = _build_manager({"audio": {"path": DUMMY_REWARDS_PATH, "name": "reward_audio", "weight": 1.0}})
        data = DataProto.from_dict(
            tensors={"responses": torch.tensor([[1, 2]])},
            non_tensors={
                "data_source": ["audio"],
                "reward_model": [{"ground_truth": "hello"}],
                "extra_info": [{}],
                "tool_extra_fields": [{"media_kind": "audio"}],
            },
        )

        with pytest.raises(KeyError, match=r"requires extra_info\['audio'\]"):
            manager.loop.run_until_complete(manager.run_single(data))

    def test_unknown_media_kind_fails_before_scoring(self):
        manager = _build_manager({"rule": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0}})
        data = _make_single_data()
        data.non_tensor_batch["tool_extra_fields"] = np.array([{"media_kind": "mesh"}], dtype=object)

        with pytest.raises(ValueError, match="Unsupported reward media kind: 'mesh'"):
            manager.loop.run_until_complete(manager.run_single(data))

    def test_visual_reward_can_consume_auxiliary_audio(self):
        manager = _build_manager(
            {
                "audiovisual": {
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_visual_with_aux_audio",
                    "weight": 1.0,
                }
            }
        )
        data = _make_single_data()
        data.non_tensor_batch["tool_extra_fields"] = np.array(
            [
                {
                    "audio": np.arange(8, dtype=np.float32),
                    "audio_sample_rate": 32_000,
                    "media_kind": "video",
                }
            ],
            dtype=object,
        )

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.9)

    def test_assemble_rm_scores_preserves_modality_layouts(self):
        visual = DataProto.from_dict(tensors={"responses": torch.zeros(2, 3, 8, 8, dtype=torch.uint8)})
        text = DataProto.from_dict(
            tensors={
                "prompts": torch.zeros(2, 2, dtype=torch.long),
                "responses": torch.zeros(2, 3, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            }
        )

        visual_scores = MultiRewardManager.assemble_rm_scores(visual, [0.25, 0.5])
        text_scores = MultiRewardManager.assemble_rm_scores(text, [0.25, 0.5])

        torch.testing.assert_close(visual_scores, torch.tensor([[0.25], [0.5]]))
        torch.testing.assert_close(text_scores, torch.tensor([[0.0, 0.0, 0.25], [0.5, 0.0, 0.0]]))

    def test_legacy_visual_manager_remains_compatible(self):
        config = _make_config({"visual": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0}})
        manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.5)
        assert not isinstance(manager, VisualRewardManager)
        assert [type(adapter) for adapter in manager._reward_adapters] == [VisualRewardAdapter, AudioRewardAdapter]
        assert MultiVisualRewardManager.assemble_rm_scores(_make_single_data(), [0.5]).shape == (1, 1)


class TestMultiRewardManagerInit:
    def test_default_reward_manager_config_resolves_to_unified_manager(self):
        config = _make_config({"a": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0}})

        assert resolve_reward_manager_cls(config) is MultiRewardManager

    def test_empty_reward_functions_uses_single_reward_compatibility_path(self):
        manager = _build_manager({})

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data("jpeg_compressibility")))

        assert result["reward_score"] != pytest.approx(0.0)

    def test_generic_manager_uses_modality_adapters(self):
        manager = _build_manager({"a": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score"}})

        assert [type(adapter) for adapter in manager._reward_adapters] == [
            TextRewardAdapter,
            VisualRewardAdapter,
            AudioRewardAdapter,
        ]

    def test_loads_sub_rewards(self):
        reward_fns = {
            "a": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 0.5},
        }
        manager = _build_manager(reward_fns)
        assert len(manager._sub_rewards) == 1
        assert manager._sub_rewards[0]["key"] == "a"
        assert manager._sub_rewards[0]["weight"] == 0.5
        assert manager._sub_rewards[0]["required"] is False

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [(True, True), (False, False), ("true", True), ("false", False)],
    )
    def test_parses_required(self, configured, expected):
        manager = _build_manager(
            {
                "a": {
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_fixed_score",
                    "required": configured,
                }
            }
        )

        assert manager._sub_rewards[0]["required"] is expected

    @pytest.mark.parametrize("configured", ["yes", 1, None])
    def test_rejects_invalid_required(self, configured):
        with pytest.raises((TypeError, ValueError), match="required"):
            _build_manager(
                {
                    "a": {
                        "path": DUMMY_REWARDS_PATH,
                        "name": "reward_fixed_score",
                        "required": configured,
                    }
                }
            )
