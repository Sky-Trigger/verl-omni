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
from verl.protocol import DataProto

from verl_omni.reward_loop import deployment as deployment_module
from verl_omni.reward_loop.deployment import (
    EngineRewardDeployment,
    MultiRewardModelManager,
    NativeRewardDeployment,
    NativeRewardExecutor,
    RewardExecutorSpec,
    _prepare_engine_config,
    accelerator_workers_enabled,
    build_engine_reward_executors,
    reward_is_enabled,
    reward_pool_is_separate,
    reward_role_required,
    streaming_reward_enabled,
    validate_reward_deployment_terms,
)
from verl_omni.reward_loop.reward_loop import (
    OmniRewardLoopManager,
    OmniRewardLoopWorker,
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
            "ocr": {"backend": "engine"},
            "pickscore": {"backend": "engine"},
        }
    )

    with pytest.raises(ValueError, match="require a parent resource pool"):
        MultiRewardModelManager(config)


def test_mixed_engine_and_native_deployments_share_one_parent_pool(monkeypatch):
    config = _config(
        {
            "ocr": {"backend": "engine", "model_path": "/models/ocr"},
            "pickscore": {
                "backend": "native",
                "adapter": "pickscore",
                "placement": {"devices": [0, 1, 2, 3]},
            },
        }
    )
    manager = object.__new__(MultiRewardModelManager)
    manager.config = config
    manager.resource_pool = SimpleNamespace(world_size=10)
    observed = {}

    def fake_split(pool, sizes):
        observed["pool"] = pool
        observed["sizes"] = sizes
        return ["engine-pool", "native-pool", "unused-pool"]

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    engine_pools, native_pool = manager._split_deployment_resource_pools(
        [("ocr", config.reward.deployments.ocr)],
        [("pickscore", config.reward.deployments.pickscore)],
        config.reward.reward_model,
    )

    assert observed == {"pool": manager.resource_pool, "sizes": [2, 4, 4]}
    assert engine_pools == {"ocr": "engine-pool"}
    assert native_pool == "native-pool"


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
            {"backend": "engine", "rollout": {"tensor_model_parallel_size": 2}, "replicas": 2},
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
                "backend": "engine",
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
            self._spec = RewardExecutorSpec(name, "engine", None, f"{name}:8000", {})

        @property
        def executor_spec(self):
            return self._spec

        def wake_up(self):
            return None

        def sleep(self):
            return None

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    monkeypatch.setattr(deployment_module, "EngineRewardDeployment", FakeEngineDeployment)

    manager = MultiRewardModelManager(config, resource_pool=parent_pool)

    assert observed == [("pickscore", "pickscore-pool"), ("ocr", "ocr-pool")]
    assert set(manager.reward_executor_specs) == {"pickscore", "ocr"}


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


def test_native_only_deployment_uses_parent_pool_and_batch_scoring():
    config = _config({"pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0]}}})

    assert reward_is_enabled(config)
    assert reward_role_required(config)
    assert not streaming_reward_enabled(config)


def test_accelerator_worker_setting_keeps_the_legacy_alias():
    config = _config()

    assert not accelerator_workers_enabled(config)

    config.reward.accelerator_workers.enabled = True
    assert accelerator_workers_enabled(config)

    config.reward.accelerator_workers.enabled = False
    config.reward.custom_reward_function.use_accelerator = True
    assert accelerator_workers_enabled(config)


def test_engine_deployment_rejects_legacy_per_deployment_pool_switch():
    config = _config({"ocr": {"backend": "engine", "enable_resource_pool": True}})

    with pytest.raises(ValueError, match="must not set enable_resource_pool"):
        MultiRewardModelManager(config, resource_pool=SimpleNamespace(world_size=1))


def test_native_pickscore_adapter_is_selected_by_default():
    deployment = NativeRewardDeployment(
        "pickscore",
        OmegaConf.create({"backend": "native", "adapter": "pickscore", "model_path": "/models/pickscore"}),
    )

    assert deployment.executor_spec.executor_config["model"] == (
        "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel"
    )
    assert deployment.executor_spec.model_path == "/models/pickscore"


def test_native_deployment_delegates_lifecycle_to_bound_workers(monkeypatch):
    calls = []

    class _RemoteMethod:
        def __init__(self, method):
            self.method = method

        def remote(self, deployment_name):
            calls.append((self.method, deployment_name))
            return None

    workers = [
        SimpleNamespace(
            wake_up_reward_model=_RemoteMethod("wake"),
            sleep_reward_model=_RemoteMethod("sleep"),
        )
        for _ in range(2)
    ]
    deployment = NativeRewardDeployment(
        "pickscore",
        OmegaConf.create({"backend": "native", "adapter": "pickscore", "model_path": "/models/pickscore"}),
    )
    deployment.bind_workers(workers)
    monkeypatch.setattr(deployment_module.ray, "get", lambda refs: refs)

    deployment.wake_up()
    deployment.sleep()

    assert calls == [
        ("wake", "pickscore"),
        ("wake", "pickscore"),
        ("sleep", "pickscore"),
        ("sleep", "pickscore"),
    ]


def test_native_deployment_can_stay_resident(monkeypatch):
    calls = []

    class _RemoteMethod:
        def __init__(self, method):
            self.method = method

        def remote(self, deployment_name):
            calls.append((self.method, deployment_name))
            return None

    worker = SimpleNamespace(
        wake_up_reward_model=_RemoteMethod("wake"),
        sleep_reward_model=_RemoteMethod("sleep"),
    )
    deployment = NativeRewardDeployment(
        "pickscore",
        OmegaConf.create(
            {"backend": "native", "offload": False, "adapter": "pickscore", "model_path": "/models/pickscore"}
        ),
    )
    deployment.bind_workers([worker])
    monkeypatch.setattr(deployment_module.ray, "get", lambda refs: refs)

    deployment.wake_up()
    deployment.sleep()
    deployment.wake_up()

    assert calls == [("wake", "pickscore")]


@pytest.mark.parametrize(
    ("deployments", "message"),
    [
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": []},
                }
            },
            "non-empty list",
        ),
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": [0, 0]},
                }
            },
            "duplicate",
        ),
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": [-1]},
                }
            },
            "non-negative integers",
        ),
        (
            {
                "pickscore": {
                    "backend": "native",
                    "adapter": "pickscore",
                    "placement": {"devices": [0]},
                    "rollout": {"tensor_model_parallel_size": 2},
                }
            },
            "does not support engine resource fields",
        ),
    ],
)
def test_native_deployment_rejects_invalid_placement(deployments, message):
    with pytest.raises(ValueError, match=message):
        MultiRewardModelManager(_config(deployments), resource_pool=SimpleNamespace(world_size=8))


def test_native_deployment_rejects_overlapping_device_assignments():
    config = _config(
        {
            "pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0, 1]}},
            "hpsv3": {
                "backend": "native",
                "executor": {"model": "tests.fake:HpsModel"},
                "placement": {"devices": [1, 2]},
            },
        }
    )

    with pytest.raises(ValueError, match="overlaps index 1"):
        MultiRewardModelManager(config, resource_pool=SimpleNamespace(world_size=8))


def test_native_device_assignments_size_the_native_subpool(monkeypatch):
    config = _config(
        {
            "pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0, 1]}},
            "hpsv3": {
                "backend": "native",
                "executor": {"model": "tests.fake:HpsModel"},
                "placement": {"devices": [4, 5]},
            },
        }
    )
    manager = object.__new__(MultiRewardModelManager)
    manager.config = config
    manager.resource_pool = SimpleNamespace(world_size=8)
    manager.native_device_assignments = manager._validate_native_device_assignments(
        [(name, deployment) for name, deployment in config.reward.deployments.items()]
    )
    observed = {}

    def fake_split(pool, sizes):
        observed["pool"] = pool
        observed["sizes"] = sizes
        return ["native-pool", "unused-pool"]

    monkeypatch.setattr(deployment_module, "split_resource_pool", fake_split)
    engine_pools, native_pool = manager._split_deployment_resource_pools(
        [], list(config.reward.deployments.items()), config.reward.reward_model
    )

    assert observed == {"pool": manager.resource_pool, "sizes": [6, 2]}
    assert engine_pools == {}
    assert native_pool == "native-pool"
    assert manager.native_device_assignments == {"pickscore": (0, 1), "hpsv3": (4, 5)}


def test_native_deployments_create_isolated_worker_groups(monkeypatch):
    config = _config(
        {
            "pickscore": {"backend": "native", "adapter": "pickscore", "placement": {"devices": [0, 1]}},
            "hpsv3": {
                "backend": "native",
                "executor": {"model": "tests.fake:HpsModel"},
                "placement": {"devices": [2, 3]},
            },
        }
    )
    config.reward.reward_functions = OmegaConf.create(
        {
            "pickscore": {
                "deployment": "pickscore",
                "path": "tests.fake.py",
                "name": "score_pickscore",
            },
            "hpsv3": {
                "deployment": "hpsv3",
                "path": "tests.fake.py",
                "name": "score_hpsv3",
            },
        }
    )
    specs = {
        "pickscore": RewardExecutorSpec("pickscore", "native", "/models/pickscore", None, {}),
        "hpsv3": RewardExecutorSpec("hpsv3", "native", "/models/hpsv3", None, {}),
    }
    manager = object.__new__(OmniRewardLoopManager)
    manager.config = config
    manager.reward_router_address = None
    manager.multi_reward_model_manager = SimpleNamespace(
        deployments={"pickscore": object(), "hpsv3": object()},
        reward_executor_specs=specs,
        native_device_assignments={"pickscore": (0, 1), "hpsv3": (2, 3)},
        bind_native_workers=lambda name, workers: observed_bindings.append((name, workers)),
    )
    observed = []
    observed_bindings = []

    def create_native_workers(group_config, group_specs, bundle_indices, name_prefix):
        observed.append((group_config, group_specs, bundle_indices, name_prefix))
        return [f"{name_prefix}-worker"]

    monkeypatch.setattr("verl_omni.reward_loop.reward_loop.ray.remote", lambda cls: cls)
    manager._create_node_affinity_workers = lambda *args: pytest.fail("no shared worker group expected")
    manager._create_native_workers = create_native_workers

    manager._init_reward_loop_workers()

    assert manager._reward_worker_groups == {
        "pickscore": ["native_reward_loop_worker_pickscore-worker"],
        "hpsv3": ["native_reward_loop_worker_hpsv3-worker"],
    }
    assert manager.reward_loop_workers == [
        "native_reward_loop_worker_pickscore-worker",
        "native_reward_loop_worker_hpsv3-worker",
    ]
    observed_groups = [
        (set(group_specs), tuple(bundle_indices), name_prefix)
        for _, group_specs, bundle_indices, name_prefix in observed
    ]
    assert observed_groups == [
        ({"pickscore"}, (0, 1), "native_reward_loop_worker_pickscore"),
        ({"hpsv3"}, (2, 3), "native_reward_loop_worker_hpsv3"),
    ]
    assert [set(group_config.reward.reward_functions) for group_config, *_ in observed] == [
        {"pickscore"},
        {"hpsv3"},
    ]
    assert observed_bindings == [
        ("pickscore", ["native_reward_loop_worker_pickscore-worker"]),
        ("hpsv3", ["native_reward_loop_worker_hpsv3-worker"]),
    ]


def test_engine_config_fills_the_default_rollout_name():
    config = _config()
    engine_config = _prepare_engine_config(
        OmegaConf.create({"backend": "engine", "model_path": "/models/clip"}),
        config.reward.reward_model,
    )

    assert engine_config.enable is True
    assert engine_config.rollout.name == "vllm"
    assert engine_config.rollout.free_cache_engine is True
    assert engine_config.rollout.enable_sleep_mode is True
    assert "backend" not in engine_config


def test_engine_offload_false_disables_vllm_sleep_mode():
    config = _config()
    engine_config = _prepare_engine_config(
        OmegaConf.create({"backend": "engine", "offload": False, "model_path": "/models/clip"}),
        config.reward.reward_model,
    )

    assert engine_config.rollout.free_cache_engine is False
    assert engine_config.rollout.enable_sleep_mode is False
    assert "offload" not in engine_config


def test_engine_resident_deployment_skips_wake_and_sleep():
    manager = SimpleNamespace(
        wake_up=lambda: pytest.fail("unexpected wake"),
        sleep=lambda: pytest.fail("unexpected sleep"),
    )
    deployment = object.__new__(EngineRewardDeployment)
    deployment.offload = False
    deployment.reward_model_manager = manager

    deployment.wake_up()
    deployment.sleep()


@pytest.mark.parametrize("offload", ["true", 1, 0])
def test_deployment_offload_must_be_boolean(offload):
    with pytest.raises(ValueError, match="offload must be a boolean"):
        NativeRewardDeployment(
            "native",
            OmegaConf.create(
                {
                    "backend": "native",
                    "offload": offload,
                    "executor": {"model": "tests.fake:Model"},
                }
            ),
        )


def test_engine_offload_rejects_conflicting_legacy_sleep_setting():
    config = _config()
    with pytest.raises(ValueError, match="conflicts with rollout sleep settings"):
        _prepare_engine_config(
            OmegaConf.create(
                {
                    "backend": "engine",
                    "offload": False,
                    "model_path": "/models/clip",
                    "rollout": {"free_cache_engine": True},
                }
            ),
            config.reward.reward_model,
        )


def test_pooling_engine_uses_pooling_safe_worker_extension():
    config = _config()
    engine_config = _prepare_engine_config(
        OmegaConf.create(
            {
                "backend": "engine",
                "model_path": "/models/clip",
                "rollout": {"name": "vllm", "engine_kwargs": {"vllm": {"runner": "pooling"}}},
            }
        ),
        config.reward.reward_model,
    )

    assert engine_config.rollout.engine_kwargs.vllm.worker_extension_cls == (
        "verl_omni.reward_loop.vllm_worker.PoolingRewardModelWorkerExtension"
    )


def test_pooling_engine_preserves_explicit_worker_extension():
    config = _config()
    engine_config = _prepare_engine_config(
        OmegaConf.create(
            {
                "backend": "engine",
                "model_path": "/models/clip",
                "rollout": {
                    "name": "vllm",
                    "engine_kwargs": {"vllm": {"runner": "pooling", "worker_extension_cls": "custom.WorkerExtension"}},
                },
            }
        ),
        config.reward.reward_model,
    )

    assert engine_config.rollout.engine_kwargs.vllm.worker_extension_cls == "custom.WorkerExtension"


def test_engine_executor_exposes_only_router_arguments():
    spec = RewardExecutorSpec(
        name="pickscore",
        backend="engine",
        model_path="/models/pickscore",
        router_address="router:8000",
        executor_config={},
    )

    executor = build_engine_reward_executors({"pickscore": spec})["pickscore"]

    assert executor.reward_kwargs() == {
        "reward_router_address": "router:8000",
        "model_name": "/models/pickscore",
    }


@pytest.mark.parametrize(
    ("deployments", "term", "message"),
    [
        ({}, {"deployment": "missing"}, "unknown deployment"),
        (
            {"native": {"backend": "native", "executor": {"model": "unused:Unused"}}},
            {"deployment": "native"},
            "needs path/name",
        ),
        (
            {"engine": {"backend": "engine"}},
            {"deployment": "engine"},
            "needs path/name",
        ),
    ],
)
def test_reward_deployment_terms_fail_fast(deployments, term, message):
    config = _config(deployments)
    config.reward.reward_functions = OmegaConf.create({"term": term})

    with pytest.raises(ValueError, match=message):
        validate_reward_deployment_terms(config)


def test_engine_pickscore_term_uses_a_reward_function():
    config = _config({"pickscore": {"backend": "engine"}})
    config.reward.reward_functions = OmegaConf.create(
        {
            "pickscore": {
                "deployment": "pickscore",
                "path": "pkg://verl_omni.utils.reward_score.pickscore_reward",
                "name": "compute_score_pickscore_engine",
                "logit_scale": 98.86447,
            }
        }
    )

    validate_reward_deployment_terms(config)


def test_native_pickscore_term_uses_a_reward_function():
    config = _config({"pickscore": {"backend": "native"}})
    config.reward.reward_functions = OmegaConf.create(
        {
            "pickscore": {
                "deployment": "pickscore",
                "path": "pkg://verl_omni.utils.reward_score.pickscore_reward",
                "name": "compute_score_pickscore_native",
            }
        }
    )

    validate_reward_deployment_terms(config)


class _FakeModel:
    instances = []

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.closed = False
        self.__class__.instances.append(self)

    def infer(self, prompts, images):
        assert prompts == ["prompt"]
        assert len(images) == 1
        return torch.tensor([0.75])

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_native_executor_wakes_infers_and_sleeps(monkeypatch):
    spec = RewardExecutorSpec(
        name="native",
        backend="native",
        model_path="/models/native",
        router_address=None,
        executor_config={"model": "tests.fake:FakeModel"},
    )
    executor = NativeRewardExecutor(spec)
    _FakeModel.instances.clear()
    monkeypatch.setattr(deployment_module, "_load_native_model", lambda _: _FakeModel)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    await executor.wake_up()
    result = await executor.infer(["prompt"], [torch.zeros(3, 2, 2, dtype=torch.uint8)])
    await executor.sleep()

    torch.testing.assert_close(result, torch.tensor([0.75]))
    assert len(_FakeModel.instances) == 1
    assert _FakeModel.instances[0].model_path == "/models/native"
    assert _FakeModel.instances[0].device == torch.device("cpu", 0)
    assert _FakeModel.instances[0].closed
    assert executor._model is None


@pytest.mark.asyncio
async def test_native_executor_rejects_inference_while_asleep():
    executor = NativeRewardExecutor(
        RewardExecutorSpec(
            name="native",
            backend="native",
            model_path=None,
            router_address=None,
            executor_config={"model": "tests.fake:FakeModel"},
        )
    )

    with pytest.raises(RuntimeError, match="is not awake"):
        await executor.infer(["prompt"], [torch.zeros(3, 2, 2, dtype=torch.uint8)])


@pytest.mark.asyncio
async def test_native_executor_waits_for_inflight_score_before_sleep(monkeypatch):
    class _BlockingModel:
        release = False
        closed = False

        def __init__(self, **kwargs):
            del kwargs

        def infer(self, prompts, images):
            del prompts, images
            import time

            while not self.release:
                time.sleep(0.01)
            return torch.tensor([0.5])

        def close(self):
            self.closed = True

    spec = RewardExecutorSpec(
        name="native",
        backend="native",
        model_path=None,
        router_address=None,
        executor_config={"model": "tests.fake:BlockingModel"},
    )
    executor = NativeRewardExecutor(spec)
    _BlockingModel.release = False
    monkeypatch.setattr(deployment_module, "_load_native_model", lambda _: _BlockingModel)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    await executor.wake_up()
    infer_task = asyncio.create_task(executor.infer(["prompt"], [torch.zeros(3, 2, 2, dtype=torch.uint8)]))
    await asyncio.sleep(0.02)
    sleep_task = asyncio.create_task(executor.sleep())
    await asyncio.sleep(0.02)
    assert not sleep_task.done()
    _BlockingModel.release = True
    torch.testing.assert_close(await infer_task, torch.tensor([0.5]))
    await sleep_task
    assert executor._model is None


@pytest.mark.asyncio
async def test_native_executor_does_not_serialize_async_model_calls(monkeypatch):
    class _BatchingModel:
        instances = []

        def __init__(self, **kwargs):
            del kwargs
            self._entered = 0
            self._second_request = asyncio.Event()
            self.closed = False
            self.__class__.instances.append(self)

        async def infer(self, prompts, images):
            del prompts, images
            self._entered += 1
            if self._entered == 2:
                self._second_request.set()
            await self._second_request.wait()
            return [0.5]

        async def close(self):
            self.closed = True

    spec = RewardExecutorSpec(
        name="native",
        backend="native",
        model_path=None,
        router_address=None,
        executor_config={"model": "tests.fake:BatchingModel"},
    )
    executor = NativeRewardExecutor(spec)
    _BatchingModel.instances.clear()
    monkeypatch.setattr(deployment_module, "_load_native_model", lambda _: _BatchingModel)
    monkeypatch.setattr(deployment_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(deployment_module, "get_device_id", lambda: 0)

    await executor.wake_up()
    results = await asyncio.wait_for(
        asyncio.gather(
            executor.infer(["first"], [torch.zeros(3, 2, 2, dtype=torch.uint8)]),
            executor.infer(["second"], [torch.zeros(3, 2, 2, dtype=torch.uint8)]),
        ),
        timeout=1,
    )
    await executor.sleep()

    assert results == [[0.5], [0.5]]
    assert _BatchingModel.instances[0]._entered == 2
    assert _BatchingModel.instances[0].closed


@pytest.mark.asyncio
async def test_worker_exposes_native_model_lifecycle():
    worker = object.__new__(OmniRewardLoopWorker)
    executor = SimpleNamespace(wake_up=AsyncMock(), sleep=AsyncMock())
    worker.native_reward_executors = {"native": executor}

    await worker.wake_up_reward_model("native")
    await worker.sleep_reward_model("native")

    executor.wake_up.assert_awaited_once_with()
    executor.sleep.assert_awaited_once_with()


def test_deployment_manager_rejects_legacy_and_named_models():
    config = _config({"pickscore": {"backend": "native", "adapter": "pickscore"}})
    config.reward.reward_model.enable = True

    with pytest.raises(ValueError, match="cannot be combined"):
        MultiRewardModelManager(config)


def test_native_deployment_requires_allocated_native_resource_pool():
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = SimpleNamespace(
        native_resource_pool=None,
    )

    with pytest.raises(ValueError, match="require an allocated native resource pool"):
        manager._create_native_workers(
            config=SimpleNamespace(),
            specs={},
            bundle_indices=[0],
            name_prefix="native_reward_loop_worker_pickscore",
        )


def test_named_deployment_groups_merge_scores_and_extra_info(monkeypatch):
    class _Worker:
        def __init__(self, outputs):
            self._outputs = outputs
            self.compute_score_batch = SimpleNamespace(remote=lambda data: self._outputs[: len(data)])

    class _RewardManager:
        @staticmethod
        def assemble_rm_scores(data, scores):
            del data
            return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    data = DataProto.from_dict(
        tensors={"responses": torch.zeros(2, 3, 2, 2, dtype=torch.uint8)},
        non_tensors={"data_source": ["a", "b"]},
    )
    manager = object.__new__(OmniRewardLoopManager)
    manager.reward_manager_cls = _RewardManager
    manager._reward_worker_groups = {
        "shared": [
            _Worker(
                [
                    {"reward_score": 0.25, "reward_extra_info": {"reward/ocr": 0.25, "reward/combined": 0.25}},
                    {"reward_score": 0.5, "reward_extra_info": {"reward/ocr": 0.5, "reward/combined": 0.5}},
                ]
            )
        ],
        "pickscore": [
            _Worker(
                [
                    {
                        "reward_score": 0.75,
                        "reward_extra_info": {"reward/pickscore": 0.75, "reward/combined": 0.75},
                    },
                    {
                        "reward_score": 1.0,
                        "reward_extra_info": {"reward/pickscore": 1.0, "reward/combined": 1.0},
                    },
                ]
            )
        ],
    }
    monkeypatch.setattr("verl_omni.reward_loop.reward_loop.ray.get", lambda refs: refs)

    result = manager._compute_named_deployment_scores(data)

    assert torch.equal(result.batch["rm_scores"], torch.tensor([[1.0], [1.5]]))
    assert result.non_tensor_batch["reward/ocr"].tolist() == [0.25, 0.5]
    assert result.non_tensor_batch["reward/pickscore"].tolist() == [0.75, 1.0]
    assert result.non_tensor_batch["reward/combined"].tolist() == [1.0, 1.5]
    assert result.meta_info["reward_extra_keys"] == ["reward/ocr", "reward/pickscore", "reward/combined"]


def test_compute_rm_score_brackets_scoring_with_one_lifecycle():
    calls = []
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = SimpleNamespace(
        wake_up=lambda: calls.append("wake_up"),
        sleep=lambda: calls.append("sleep"),
    )
    manager._reward_worker_groups = {"engine": [object()], "native": [object()]}
    manager._compute_named_deployment_scores = lambda data: calls.append("score") or data

    data = object()
    assert manager.compute_rm_score(data) is data
    assert calls == ["wake_up", "score", "sleep"]
