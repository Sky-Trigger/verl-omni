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
import asyncio

import ray
from omegaconf import open_dict
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.trainer.ppo.reward import resolve_reward_manager_cls

from .deployment import (
    MultiRewardModelManager,
    NativeRewardExecutor,
    PickScoreEngineAdapter,
    RewardDeploymentSpec,
    RouterEngineRewardClient,
    build_native_executor,
    has_reward_deployments,
    is_engine_backend,
    streaming_reward_enabled,
    validate_reward_deployment_terms,
)


def _build_deployment_clients(specs: dict[str, RewardDeploymentSpec]):
    clients = {}
    for name, spec in specs.items():
        if not is_engine_backend(spec.backend):
            continue
        adapter = spec.native.get("adapter")
        if adapter == "pickscore":
            if "logit_scale" not in spec.native:
                raise ValueError(
                    f"PickScore engine deployment {name!r} requires native.logit_scale; "
                    "vLLM does not load the checkpoint logit_scale."
                )
            clients[name] = PickScoreEngineAdapter(
                router_address=spec.router_address,
                model_path=spec.model_path,
                logit_scale=float(spec.native["logit_scale"]),
                score_divisor=float(spec.native.get("score_divisor", 26.0)),
            )
        elif adapter is None:
            clients[name] = RouterEngineRewardClient(
                router_address=spec.router_address,
                model_path=spec.model_path,
            )
        else:
            raise ValueError(f"Unsupported engine reward adapter {adapter!r} for deployment {name!r}")
    return clients


class OmniRewardLoopWorker(RewardLoopWorker):
    """RewardLoopWorker with named engine clients and native model executors."""

    def __init__(
        self,
        config,
        reward_router_address=None,
        reward_deployment_specs=None,
    ):
        self.reward_deployment_specs = reward_deployment_specs or {}
        self.reward_deployment_clients = _build_deployment_clients(self.reward_deployment_specs)
        self.native_reward_executor: NativeRewardExecutor | None = (
            build_native_executor(self.reward_deployment_specs) if self.reward_deployment_specs else None
        )
        self._native_batch_active = False
        super().__init__(config, reward_router_address)

    def _init_reward_fn(self):
        super()._init_reward_fn()
        if hasattr(self.reward_manager, "set_deployment_context"):
            self.reward_manager.set_deployment_context(
                self.reward_deployment_clients,
                self.native_reward_executor,
            )

    async def compute_score_batch(self, data):
        self._native_batch_active = True
        try:
            return await super().compute_score_batch(data)
        finally:
            self._native_batch_active = False
            if self.native_reward_executor is not None:
                await self.native_reward_executor.sleep()

    async def compute_score(self, data):
        # Streaming agent loops submit one item at a time rather than calling
        # ``compute_score_batch``.  Keep native models bounded to that request
        # in this path; otherwise a worker-local CLIP would stay resident into
        # the subsequent actor backward phase.
        if self.native_reward_executor is None or self._native_batch_active:
            return await super().compute_score(data)
        try:
            return await super().compute_score(data)
        finally:
            await self.native_reward_executor.sleep()

    async def close(self):
        if self.native_reward_executor is not None:
            await self.native_reward_executor.sleep()


class OmniRewardLoopManager(RewardLoopManager):
    """RewardLoopManager that can start/stop the profiler on the reward-model rollout servers.

    The reward-model servers are the same ``RolloutReplica`` stack as the actor rollout
    servers, whose per-server profiler fan-out already exists (``RolloutReplica.start_profile``);
    upstream ``RewardLoopManager`` just exposes no caller for it. The trainer invokes these
    around the phase where the servers actually score: the generation phase when reward
    computation streams with rollout, or ``compute_rm_score`` in colocate mode. Configured
    via ``reward.reward_model.rollout.profiler``.
    """

    def __init__(self, config, rm_resource_pool=None, accelerator_resource_pool=None):
        self.accelerator_resource_pool = accelerator_resource_pool
        if has_reward_deployments(config):
            validate_reward_deployment_terms(config)
        self.reward_deployment_manager = MultiRewardModelManager(
            config,
            # The trainer maps Role.RewardModel to global_pool or reward_pool.
            # Each engine deployment receives a sub-pool from this one parent.
            resource_pool=rm_resource_pool,
        )
        if self.reward_deployment_manager.deployments:
            if config.reward.reward_model.get("enable", False):
                raise ValueError(
                    "Use reward.deployments for deployment-backed rewards; "
                    "reward.reward_model.enable cannot be combined with reward.deployments."
                )
            if not config.reward.get("reward_functions"):
                raise ValueError("reward.deployments requires non-empty reward.reward_functions")
            self.config = config
            with open_dict(config.reward.reward_manager):
                config.reward.reward_manager.name = "MultiVisualRewardManager"
            self.reward_model_manager = None
            self.reward_router_address = None
            self.reward_loop_workers_class = ray.remote(OmniRewardLoopWorker)
            self.reward_manager_cls = resolve_reward_manager_cls(config)
            self._init_reward_loop_workers()
        else:
            super().__init__(config=config, rm_resource_pool=rm_resource_pool)

    @property
    def reward_loop_worker_handles(self):
        if not streaming_reward_enabled(self.config):
            return None
        return super().reward_loop_worker_handles

    def _init_reward_loop_workers(self):
        self.reward_loop_workers_class = ray.remote(OmniRewardLoopWorker)
        specs = self.reward_deployment_manager.worker_specs
        if specs and not any(spec.backend == "native" for spec in specs.values()):
            node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
            self.reward_loop_workers = [
                self.reward_loop_workers_class.options(
                    name=f"reward_loop_worker_{index}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_ids[index % len(node_ids)], soft=True
                    ),
                ).remote(self.config, self.reward_router_address, specs)
                for index in range(self.config.reward.num_workers)
            ]
            return
        has_native = any(spec.backend == "native" for spec in specs.values())
        if has_native:
            if self.accelerator_resource_pool is None:
                raise ValueError("Native reward deployments require an accelerator resource pool")
            from .local_accelerator_reward_loop import build_accelerator_reward_workers

            self.reward_loop_workers = build_accelerator_reward_workers(
                self,
                self.accelerator_resource_pool,
                self.reward_deployment_manager.worker_specs,
            )
            return
        super()._init_reward_loop_workers()

    def compute_rm_score(self, data):
        self.reward_deployment_manager.wake_up()
        try:
            return super().compute_rm_score(data)
        finally:
            self.reward_deployment_manager.sleep()

    def start_profile(self, **kwargs) -> None:
        """Start profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("start_profile", **kwargs)

    def stop_profile(self) -> None:
        """Stop profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("stop_profile")

    def _run_on_replicas(self, method: str, **kwargs) -> None:
        if self.reward_model_manager is None:
            return
        replicas = self.reward_model_manager.rollout_replicas

        async def run_all():
            await asyncio.gather(*[getattr(replica, method)(**kwargs) for replica in replicas])

        asyncio.run(run_all())
