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
"""CPU contracts for named reward deployments and their lifecycle."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl_omni.reward_loop import deployment as deployment_module
from verl_omni.reward_loop.deployment import (
    MultiRewardModelManager,
    NativeRewardDeployment,
    NativeRewardExecutor,
    PickScoreEngineAdapter,
    RewardDeploymentManager,
    RewardDeploymentSpec,
    _prepare_engine_config,
    reward_is_enabled,
    reward_pool_is_separate,
    reward_role_required,
    streaming_reward_enabled,
    validate_reward_deployment_terms,
)
from verl_omni.reward_loop.reward_loop import (
    OmniRewardLoopManager,
    OmniRewardLoopWorker,
    _build_deployment_clients,
)


def _config(deployments=None):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.reward_model.enable = False
    config.reward.deployments = OmegaConf.create(deployments or {})
    return config


def test_engine_deployments_require_parent_pool():
    config = _config(
        {
            "ocr": {"backend": "verl_engine"},
            "pickscore": {"backend": "engine"},
        }
    )

    with pytest.raises(ValueError, match="require a parent resource pool"):
        RewardDeploymentManager(config)


def test_mixed_engine_and_native_deployments_require_separate_reward_pool():
    config = _config(
        {
            "ocr": {"backend": "verl_engine", "model_path": "/models/ocr"},
            "pickscore": {"backend": "native", "adapter": "pickscore"},
        }
    )

    with pytest.raises(ValueError, match="Mixed engine and native"):
        MultiRewardModelManager(config, resource_pool=SimpleNamespace(world_size=1))

    config.reward.reward_model.enable_resource_pool = True
    with pytest.raises(ValueError, match="requires a trainer-selected parent resource pool"):
        MultiRewardModelManager(config)


def test_multi_reward_model_manager_splits_parent_pool(monkeypatch):
    manager = object.__new__(MultiRewardModelManager)
    manager.resource_pool = SimpleNamespace(world_size=8)
    parent = manager.resource_pool
    observed = {}

    def fake_split(pool, sizes):
        observed["pool"] = pool
        observed["sizes"] = sizes
        return [f"sub-{index}" for index in range(len(sizes))]

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    entries = [
        (
            "pickscore",
            {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 2},
        ),
        (
            "ocr",
            {"backend": "verl_engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 2},
        ),
    ]
    base_config = {"rollout": {"tensor_model_parallel_size": 1}}

    result = manager._split_engine_resource_pool(entries, base_config)

    assert observed == {"pool": parent, "sizes": [4, 4]}
    assert result == {"pickscore": "sub-0", "ocr": "sub-1"}


def test_multi_reward_model_manager_rejects_parent_pool_overcommit():
    manager = object.__new__(MultiRewardModelManager)
    manager.resource_pool = SimpleNamespace(world_size=4)
    entries = [
        ("one", {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 2}),
        ("two", {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 1}),
    ]

    with pytest.raises(ValueError, match="request 6 devices"):
        manager._split_engine_resource_pool(entries, {"rollout": {"tensor_model_parallel_size": 1}})


def test_multi_reward_model_manager_binds_each_engine_to_its_sub_pool(monkeypatch):
    config = _config(
        {
            "pickscore": {
                "backend": "engine",
                "model_path": "/models/pickscore",
                "rollout": {"tensor_model_parallel_size": 2},
            },
            "ocr": {
                "backend": "verl_engine",
                "model_path": "/models/ocr",
                "rollout": {"tensor_model_parallel_size": 2},
            },
        }
    )
    parent_pool = SimpleNamespace(world_size=4)
    observed = []

    def fake_split(pool, sizes):
        assert pool is parent_pool
        assert sizes == [2, 2]
        return ["pickscore-pool", "ocr-pool"]

    class FakeEngineDeployment:
        def __init__(self, name, deployment, base_config, resource_pool, fallback_model):
            del deployment, base_config, fallback_model
            observed.append((name, resource_pool))
            self._spec = RewardDeploymentSpec(name, "verl_engine", None, f"{name}:8000", {})

        @property
        def worker_spec(self):
            return self._spec

        def wake_up(self):
            return None

        def sleep(self):
            return None

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    monkeypatch.setattr(deployment_module, "VerlEngineRewardDeployment", FakeEngineDeployment)

    manager = MultiRewardModelManager(config, resource_pool=parent_pool)

    assert observed == [("pickscore", "pickscore-pool"), ("ocr", "ocr-pool")]
    assert set(manager.worker_specs) == {"pickscore", "ocr"}


def test_engine_deployment_requires_trainer_parent_pool():
    config = _config({"pickscore": {"backend": "engine"}})
    assert reward_is_enabled(config)
    assert reward_role_required(config)
    assert not reward_pool_is_separate(config)
    assert not streaming_reward_enabled(config)

    config.reward.reward_model.enable_resource_pool = True
    assert reward_role_required(config)
    assert reward_pool_is_separate(config)
    assert not streaming_reward_enabled(config)


def test_native_only_deployment_does_not_create_a_reward_model_pool():
    config = _config({"pickscore": {"backend": "native", "adapter": "pickscore"}})

    assert reward_is_enabled(config)
    assert not reward_role_required(config)
    assert streaming_reward_enabled(config)


def test_engine_deployment_rejects_legacy_per_deployment_pool_switch():
    config = _config({"ocr": {"backend": "engine", "enable_resource_pool": True}})

    with pytest.raises(ValueError, match="must not set enable_resource_pool"):
        MultiRewardModelManager(config, resource_pool=SimpleNamespace(world_size=1))


def test_native_pickscore_adapter_is_selected_by_default():
    deployment = NativeRewardDeployment(
        "pickscore",
        OmegaConf.create({"backend": "native", "adapter": "pickscore", "model_path": "/models/pickscore"}),
    )

    assert deployment.worker_spec.native["scorer"] == (
        "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeScorer"
    )
    assert deployment.worker_spec.model_path == "/models/pickscore"


def test_engine_config_fills_the_default_rollout_name():
    config = _config()
    engine_config = _prepare_engine_config(
        OmegaConf.create({"backend": "verl_engine", "model_path": "/models/clip"}),
        config.reward.reward_model,
    )

    assert engine_config.enable is True
    assert engine_config.rollout.name == "vllm"
    assert "backend" not in engine_config


def test_pickscore_engine_requires_explicit_logit_scale():
    spec = RewardDeploymentSpec(
        name="pickscore",
        backend="verl_engine",
        model_path="/models/pickscore",
        router_address="router:8000",
        native={"adapter": "pickscore"},
    )

    with pytest.raises(ValueError, match="requires native.logit_scale"):
        _build_deployment_clients({"pickscore": spec})


@pytest.mark.parametrize(
    ("deployments", "term", "message"),
    [
        ({}, {"deployment": "missing"}, "unknown deployment"),
        (
            {"native": {"backend": "native", "native": {"scorer": "unused:Unused"}}},
            {"deployment": "native", "path": "tests.fake.py", "name": "score"},
            "scores directly",
        ),
        (
            {"engine": {"backend": "verl_engine"}},
            {"deployment": "engine"},
            "needs path/name",
        ),
        (
            {"pickscore": {"backend": "verl_engine", "adapter": "pickscore"}},
            {"deployment": "pickscore", "path": "tests.fake.py", "name": "score"},
            "scores directly",
        ),
    ],
)
def test_reward_deployment_terms_fail_fast(deployments, term, message):
    config = _config(deployments)
    config.reward.reward_functions = OmegaConf.create({"term": term})

    with pytest.raises(ValueError, match=message):
        validate_reward_deployment_terms(config)


class _FakeScorer:
    instances = []

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.closed = False
        self.__class__.instances.append(self)

    def score(self, prompts, images):
        assert prompts == ["prompt"]
        assert len(images) == 1
        return torch.tensor([0.75])

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_native_executor_wakes_scores_and_sleeps(monkeypatch):
    spec = RewardDeploymentSpec(
        name="native",
        backend="native",
        model_path="/models/native",
        router_address=None,
        native={"scorer": "tests.fake:FakeScorer"},
    )
    executor = NativeRewardExecutor({"native": spec})
    _FakeScorer.instances.clear()
    monkeypatch.setattr(deployment_module, "_load_native_scorer", lambda _: _FakeScorer)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    await executor.wake_up("native")
    result = await executor.score("native", "prompt", torch.zeros(3, 2, 2, dtype=torch.uint8))
    await executor.sleep()

    assert result == {"score": 0.75, "pickscore_raw": 0.75}
    assert len(_FakeScorer.instances) == 1
    assert _FakeScorer.instances[0].model_path == "/models/native"
    assert _FakeScorer.instances[0].device == torch.device("cpu", 0)
    assert _FakeScorer.instances[0].closed
    assert executor._scorers == {}


@pytest.mark.asyncio
async def test_native_executor_waits_for_inflight_score_before_sleep(monkeypatch):
    class _BlockingScorer:
        release = False
        closed = False

        def __init__(self, **kwargs):
            del kwargs

        def score(self, prompts, images):
            del prompts, images
            import time

            while not self.release:
                time.sleep(0.01)
            return torch.tensor([0.5])

        def close(self):
            self.closed = True

    spec = RewardDeploymentSpec(
        name="native",
        backend="native",
        model_path=None,
        router_address=None,
        native={"scorer": "tests.fake:BlockingScorer"},
    )
    executor = NativeRewardExecutor({"native": spec})
    _BlockingScorer.release = False
    monkeypatch.setattr(deployment_module, "_load_native_scorer", lambda _: _BlockingScorer)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    score_task = asyncio.create_task(executor.score("native", "prompt", torch.zeros(3, 2, 2, dtype=torch.uint8)))
    await asyncio.sleep(0.02)
    sleep_task = asyncio.create_task(executor.sleep())
    await asyncio.sleep(0.02)
    assert not sleep_task.done()
    _BlockingScorer.release = True
    assert await score_task == {"score": 0.5, "pickscore_raw": 0.5}
    await sleep_task
    assert executor._scorers == {}


@pytest.mark.asyncio
async def test_pickscore_engine_adapter_posts_openai_embedding_payloads(monkeypatch):
    requests = []

    class _Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return {"data": [{"embedding": [1.0, 0.0]}]}

    class _Session:
        def __init__(self, **kwargs):
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, url, json):
            requests.append((url, json))
            return _Response()

    monkeypatch.setattr(deployment_module.aiohttp, "ClientSession", _Session)
    adapter = PickScoreEngineAdapter("router:8000", "pickscore", logit_scale=98.0, score_divisor=26.0)

    result = await adapter.score("prompt", torch.zeros(3, 2, 2, dtype=torch.uint8))

    assert result == {"score": pytest.approx(98.0 / 26.0), "pickscore_raw": pytest.approx(98.0 / 26.0)}
    assert [url for url, _ in requests] == ["http://router:8000/v1/embeddings"] * 2
    assert requests[0][1] == {"model": "pickscore", "input": "prompt", "encoding_format": "float"}
    assert requests[1][1]["encoding_format"] == "float"
    assert requests[1][1]["input"][0]["content"][0]["type"] == "image_url"


@pytest.mark.asyncio
async def test_streaming_worker_sleeps_native_models_after_one_request(monkeypatch):
    worker = object.__new__(OmniRewardLoopWorker)
    executor = SimpleNamespace(sleep=AsyncMock())
    worker.native_reward_executor = executor
    worker._native_batch_active = False

    async def compute_score(_self, data):
        return {"reward_score": data}

    monkeypatch.setattr(
        "verl.experimental.reward_loop.reward_loop.RewardLoopWorker.compute_score",
        compute_score,
    )

    assert await worker.compute_score("streaming") == {"reward_score": "streaming"}
    executor.sleep.assert_awaited_once_with()


def test_deployment_manager_rejects_legacy_and_named_models():
    config = _config({"pickscore": {"backend": "native", "adapter": "pickscore"}})
    config.reward.reward_model.enable = True

    with pytest.raises(ValueError, match="cannot be combined"):
        RewardDeploymentManager(config)


def test_native_deployment_requires_accelerator_resource_pool():
    manager = object.__new__(OmniRewardLoopManager)
    manager.accelerator_resource_pool = None
    manager.reward_deployment_manager = SimpleNamespace(
        worker_specs={
            "pickscore": RewardDeploymentSpec(
                name="pickscore",
                backend="native",
                model_path="/models/pickscore",
                router_address=None,
                native={"scorer": "tests.fake:FakeScorer"},
            )
        }
    )

    with pytest.raises(ValueError, match="require an accelerator resource pool"):
        manager._init_reward_loop_workers()
