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
"""Managed reward-model deployments used by :mod:`verl_omni.reward_loop`.

The upstream ``RewardModelManager`` remains the owner of a regular verl reward
model.  This module puts that manager, arbitrary engine endpoints, and local
native models behind the same lifecycle boundary so a visual reward can select
the deployment it needs without making ``RewardLoopWorker`` own all models.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import importlib
import inspect
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import aiohttp
import torch
from omegaconf import OmegaConf
from PIL import Image
from verl.experimental.reward_loop.reward_model import RewardModelManager
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)

_ENGINE_BACKENDS = {"verl_engine", "engine"}
_NATIVE_BACKENDS = {"native"}
_ENGINE_RESOURCE_POOL_PREFIX = "reward_deployment_"


def has_reward_deployments(config) -> bool:
    """Return whether the new deployment directory has at least one entry."""
    deployments = config.reward.get("deployments", {})
    return bool(deployments)


def reward_is_enabled(config) -> bool:
    """Return whether either legacy or deployment reward computation is enabled."""
    reward_model = config.reward.get("reward_model", {})
    return bool(reward_model.get("enable", False) or has_reward_deployments(config))


def has_engine_reward_deployments(config) -> bool:
    return any(
        is_engine_backend(deployment.get("backend"))
        for deployment in (config.reward.get("deployments", {}) or {}).values()
    )


def is_engine_backend(backend: str | None) -> bool:
    return backend in _ENGINE_BACKENDS


def has_native_reward_deployments(config) -> bool:
    return any(
        deployment.get("backend") in _NATIVE_BACKENDS
        for deployment in (config.reward.get("deployments", {}) or {}).values()
    )


def validate_reward_deployment_terms(config) -> None:
    """Check that multi-reward terms and their named deployments agree.

    Validate before any ``RewardModelManager`` is constructed: a typo in a
    term must not first allocate an engine replica group and only then fail in
    a worker. Native scorers own their whole score operation, while a generic
    engine deployment needs the term's existing function to turn its router
    response into a reward. ``pickscore`` is the engine adapter currently
    supplied by verl-omni and therefore scores directly.
    """
    deployments = config.reward.get("deployments", {}) or {}
    for term_name, term in (config.reward.get("reward_functions", {}) or {}).items():
        deployment_name = term.get("deployment")
        if deployment_name is None:
            continue
        if deployment_name not in deployments:
            raise ValueError(f"Reward term {term_name!r} references unknown deployment {deployment_name!r}")

        deployment = deployments[deployment_name]
        has_function = term.get("path") is not None
        backend = deployment.get("backend")
        adapter = deployment.get("adapter") or _coerce_mapping(deployment.get("native")).get("adapter")
        if backend in _NATIVE_BACKENDS and has_function:
            raise ValueError(
                f"Native reward deployment {deployment_name!r} scores directly; "
                f"remove path/name from reward term {term_name!r}"
            )
        if is_engine_backend(backend) and adapter is None and not has_function:
            raise ValueError(
                f"Engine reward deployment {deployment_name!r} needs path/name in reward term {term_name!r} "
                "to consume its router"
            )
        if is_engine_backend(backend) and adapter == "pickscore" and has_function:
            raise ValueError(
                f"PickScore engine deployment {deployment_name!r} scores directly; "
                f"remove path/name from reward term {term_name!r}"
            )


def reward_role_required(config) -> bool:
    """Whether the legacy reward model needs the ``Role.RewardModel`` pool.

    Named engine deployments own their resource pools directly.  They are not
    a worker role, so registering them through the single legacy role would
    make two independent deployments share one set of GPUs.
    """
    return bool(config.reward.reward_model.get("enable", False))


def reward_pool_is_separate(config) -> bool:
    """Whether the legacy reward-model role uses a dedicated resource pool."""
    return bool(config.reward.reward_model.get("enable_resource_pool", False))


def streaming_reward_enabled(config) -> bool:
    """Whether workers can score while the agent rollout is still streaming."""
    if not reward_is_enabled(config):
        return True
    if has_engine_reward_deployments(config):
        # The controller owns engine wake/sleep around ``compute_rm_score``;
        # streaming workers have no controller callback at request time.
        return False
    return has_native_reward_deployments(config) or bool(config.reward.reward_model.get("enable_resource_pool", False))


def engine_resource_pool_name(deployment_name: str) -> str:
    """Return the dedicated Ray resource-pool name for one deployment."""
    return f"{_ENGINE_RESOURCE_POOL_PREFIX}{deployment_name}"


def get_engine_deployment_resource_specs(config) -> dict[str, list[int]]:
    """Return one dedicated Ray-pool specification for each standalone engine.

    ``RewardModelManager`` starts a replica group immediately.  Giving two
    differently named deployments the legacy ``reward_pool`` would therefore
    start both groups on the same GPUs.  Keep their pools independent instead.
    """
    entries = config.reward.get("deployments", {}) or {}
    colocated = [
        name
        for name, deployment in entries.items()
        if is_engine_backend(deployment.get("backend")) and not deployment.get("enable_resource_pool", False)
    ]
    if len(colocated) > 1:
        raise ValueError(
            "At most one engine reward deployment may be colocated with actor/rollout; "
            "set enable_resource_pool=true for the other deployments."
        )
    if colocated and has_native_reward_deployments(config):
        raise ValueError(
            "A colocated engine reward deployment cannot be combined with native reward deployments; "
            "set enable_resource_pool=true for the engine deployment."
        )

    specs = {}
    for name, deployment in entries.items():
        if not is_engine_backend(deployment.get("backend")) or not deployment.get("enable_resource_pool", False):
            continue
        gpus = int(deployment.get("n_gpus_per_node", config.reward.reward_model.n_gpus_per_node))
        nnodes = int(deployment.get("nnodes", config.reward.reward_model.nnodes))
        if gpus <= 0:
            raise ValueError(f"Engine reward deployment {name!r} requires n_gpus_per_node > 0")
        if nnodes <= 0:
            raise ValueError(f"Engine reward deployment {name!r} requires nnodes > 0")
        specs[name] = [gpus] * nnodes
    return specs


def get_engine_deployment_resource_pools(config, resource_pool_manager) -> dict[str, Any]:
    """Resolve standalone deployment names to their dedicated Ray pools."""
    return {
        name: resource_pool_manager.resource_pool_dict[engine_resource_pool_name(name)]
        for name in get_engine_deployment_resource_specs(config)
    }


def uses_deployment_resource_pool(config) -> bool:
    """Return whether an engine deployment needs a separate Ray pool."""
    for deployment in (config.reward.get("deployments") or {}).values():
        if is_engine_backend(deployment.get("backend")) and deployment.get("enable_resource_pool", False):
            return True
    return False


def uses_colocated_deployment(config) -> bool:
    """Return whether an engine deployment shares the actor/rollout pool."""
    for deployment in (config.reward.get("deployments") or {}).values():
        if is_engine_backend(deployment.get("backend")) and not deployment.get("enable_resource_pool", False):
            return True
    return False


def _coerce_mapping(value) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return OmegaConf.to_container(value, resolve=False)


def _load_native_scorer(scorer_path: str):
    """Load a native scorer from ``module:Class`` or an existing file path.

    The public configuration uses Python module names so the same deployment
    works in every Ray worker without depending on a controller-local relative
    path.  File paths remain supported through verl's existing loader.
    """
    module_path, class_name = scorer_path.rsplit(":", 1)
    if module_path.startswith("pkg://"):
        module_path = module_path[len("pkg://") :].replace("/", ".")
    if "/" not in module_path and not module_path.endswith(".py"):
        return getattr(importlib.import_module(module_path), class_name)

    from verl.utils.import_utils import load_extern_object

    return load_extern_object(module_path=module_path, object_name=class_name)


def _empty_accelerator_cache() -> None:
    """Release cached allocations for the current supported accelerator."""
    accelerator = getattr(torch, get_device_name(), None)
    empty_cache = getattr(accelerator, "empty_cache", None)
    if callable(empty_cache) and getattr(accelerator, "is_available", lambda: False)():
        empty_cache()


def _prepare_engine_config(deployment, base_config, fallback_model=None):
    """Build the exact config expected by upstream ``RewardModelManager``."""
    config = OmegaConf.merge(
        OmegaConf.create(_coerce_mapping(base_config)),
        OmegaConf.create(_coerce_mapping(deployment)),
    )
    for key in ("backend", "native", "name", "enable_resource_pool", "adapter"):
        if key in config:
            del config[key]
    config.enable = True
    if config.get("model_path") is None:
        config.model_path = fallback_model
    if config.get("model_path") is None:
        raise ValueError("Engine reward deployment requires model_path")
    if config.get("rollout") is None:
        raise ValueError("Engine reward deployment requires rollout config")
    if OmegaConf.is_missing(config.rollout, "name") or config.rollout.get("name") == "???":
        config.rollout.name = "vllm"
    return config


@dataclass(frozen=True)
class RewardDeploymentSpec:
    """Static metadata shared with every reward-loop worker."""

    name: str
    backend: str
    model_path: str | None
    router_address: str | None
    native: dict[str, Any]


class RewardDeployment(ABC):
    """A model deployment with an explicit wake/score/sleep lifecycle."""

    def __init__(self, spec: RewardDeploymentSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def worker_spec(self) -> RewardDeploymentSpec:
        return self.spec

    @abstractmethod
    def wake_up(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def sleep(self) -> None:
        raise NotImplementedError


class VerlEngineRewardDeployment(RewardDeployment):
    """Engine deployment owned by the existing ``verl.RewardModelManager``."""

    def __init__(self, name: str, deployment, base_config, resource_pool, fallback_model=None):
        config = _prepare_engine_config(deployment, base_config, fallback_model)
        native = _coerce_mapping(deployment.get("native"))
        self.reward_model_manager = RewardModelManager(config, resource_pool)
        super().__init__(
            RewardDeploymentSpec(
                name=name,
                backend="verl_engine",
                model_path=config.model_path,
                router_address=self.reward_model_manager.get_router_address(),
                native={
                    **native,
                    "adapter": deployment.get("adapter") or native.get("adapter"),
                },
            )
        )

    def wake_up(self) -> None:
        self.reward_model_manager.wake_up()

    def sleep(self) -> None:
        self.reward_model_manager.sleep()


class NativeRewardDeployment(RewardDeployment):
    """Native model state that is owned by every accelerator reward worker."""

    def __init__(self, name: str, deployment):
        native = _coerce_mapping(deployment.get("native"))
        scorer = native.get("scorer")
        adapter = deployment.get("adapter") or native.get("adapter")
        if scorer is None and adapter == "pickscore":
            scorer = "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeScorer"
            native["scorer"] = scorer
        if not scorer:
            raise ValueError(f"Native reward deployment {name!r} requires native.scorer")
        super().__init__(
            RewardDeploymentSpec(
                name=name,
                backend="native",
                model_path=deployment.get("model_path"),
                router_address=None,
                native=native,
            )
        )

    def wake_up(self) -> None:
        return None

    def sleep(self) -> None:
        return None


class RewardDeploymentManager:
    """Create and lifecycle-manage all configured reward deployments.

    The manager lives in the trainer/controller process.  Engine deployments
    own their replicas through ``RewardModelManager``; native deployments are
    represented here and instantiated lazily by ``NativeRewardExecutor`` in
    the Ray reward-loop worker that owns the assigned accelerator.
    """

    def __init__(self, config, engine_resource_pool=None, engine_resource_pools=None):
        if has_reward_deployments(config) and config.reward.reward_model.get("enable", False):
            raise ValueError("reward.reward_model.enable cannot be combined with reward.deployments")
        get_engine_deployment_resource_specs(config)
        self.config = config
        self.engine_resource_pools = engine_resource_pools or {}
        self.deployments: dict[str, RewardDeployment] = {}
        entries = config.reward.get("deployments", {}) or {}
        fallback_model = config.reward.reward_model.get("model_path")
        base_config = config.reward.reward_model
        for name, deployment in entries.items():
            backend = deployment.get("backend")
            if is_engine_backend(backend):
                if deployment.get("enable_resource_pool", False):
                    resource_pool = self.engine_resource_pools.get(name)
                    if resource_pool is None:
                        raise ValueError(f"Missing resource pool for engine reward deployment {name!r}")
                else:
                    resource_pool = engine_resource_pool
                self.deployments[name] = VerlEngineRewardDeployment(
                    name, deployment, base_config, resource_pool, fallback_model
                )
            elif backend in _NATIVE_BACKENDS:
                self.deployments[name] = NativeRewardDeployment(name, deployment)
            else:
                supported = ", ".join(sorted(_ENGINE_BACKENDS | _NATIVE_BACKENDS))
                raise ValueError(
                    f"Reward deployment {name!r} has unsupported backend {backend!r}; expected one of {supported}"
                )

    @property
    def worker_specs(self) -> dict[str, RewardDeploymentSpec]:
        return {name: deployment.worker_spec for name, deployment in self.deployments.items()}

    @property
    def has_colocated_engine_deployment(self) -> bool:
        for name, deployment in (self.config.reward.get("deployments") or {}).items():
            if name in self.deployments and is_engine_backend(deployment.get("backend")):
                if not deployment.get("enable_resource_pool", False):
                    return True
        return False

    @property
    def has_engine_deployment(self) -> bool:
        return any(isinstance(deployment, VerlEngineRewardDeployment) for deployment in self.deployments.values())

    def get_worker_spec(self, name: str) -> RewardDeploymentSpec:
        try:
            return self.worker_specs[name]
        except KeyError as exc:
            raise ValueError(f"Unknown reward deployment {name!r}") from exc

    def wake_up(self) -> None:
        for deployment in self.deployments.values():
            deployment.wake_up()

    def sleep(self) -> None:
        errors = []
        for deployment in reversed(list(self.deployments.values())):
            try:
                deployment.sleep()
            except Exception as exc:  # pragma: no cover - only relevant to live engine failures
                errors.append(exc)
                logger.exception("Failed to sleep reward deployment %s", deployment.name)
        if errors:
            raise errors[0]


def _to_pil_image(image) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image


def _image_data_url(image) -> str:
    image = _to_pil_image(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class PickScoreEngineAdapter:
    """Use a vLLM CLIP embedding endpoint to calculate the PickScore formula.

    vLLM's ``CLIPEmbeddingModel`` intentionally ignores the checkpoint
    ``logit_scale`` parameter.  Keep the scale explicit in the deployment
    configuration rather than claiming that an arbitrary PickScore checkpoint
    is numerically identical to the Transformers implementation.
    """

    def __init__(self, router_address: str, model_path: str, logit_scale: float, score_divisor: float = 26.0):
        self.router_address = router_address
        self.model_path = model_path
        self.logit_scale = logit_scale
        self.score_divisor = score_divisor

    async def score(self, prompt: str, image) -> dict[str, float]:
        image_url = _image_data_url(image)
        text_payload = {
            "model": self.model_path,
            "input": prompt,
            "encoding_format": "float",
        }
        image_payload = {
            "model": self.model_path,
            "input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}],
            "encoding_format": "float",
        }
        url = f"http://{self.router_address}/v1/embeddings"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.post(url, json=text_payload) as response:
                response.raise_for_status()
                text_result = await response.json()
            async with session.post(url, json=image_payload) as response:
                response.raise_for_status()
                image_result = await response.json()
        text = torch.tensor(text_result["data"][0]["embedding"], dtype=torch.float32)
        image_embedding = torch.tensor(image_result["data"][0]["embedding"], dtype=torch.float32)
        cosine = torch.nn.functional.cosine_similarity(text.unsqueeze(0), image_embedding.unsqueeze(0)).item()
        raw_score = self.logit_scale * cosine / self.score_divisor
        return {"score": raw_score, "pickscore_raw": raw_score}


class RouterEngineRewardClient:
    """Expose one named engine router to an existing reward function.

    Many engine-backed rewards already have their task-specific logic in a
    regular reward function (for example, OCR calls chat completions and then
    applies its own string metric).  This lightweight client lets that function
    select a named deployment instead of reading the former global router.
    """

    def __init__(self, router_address: str, model_path: str):
        self.router_address = router_address
        self.model_path = model_path

    def reward_kwargs(self) -> dict[str, str]:
        return {
            "reward_router_address": self.router_address,
            "model_name": self.model_path,
        }


class NativeRewardExecutor:
    """Worker-local native-model owner with safe wake/sleep semantics."""

    def __init__(self, specs: dict[str, RewardDeploymentSpec]):
        self.specs = specs
        self._scorers: dict[str, Any] = {}
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()
        # A scorer owns one framework model instance.  ``RewardLoopWorker``
        # fans a batch out into coroutines, so protect that instance from
        # simultaneous framework calls while still allowing all callers to be
        # counted by ``sleep``.
        self._score_lock = asyncio.Lock()

    async def wake_up(self, name: str) -> None:
        async with self._lock:
            await self._wake_up_locked(name)

    async def _wake_up_locked(self, name: str) -> None:
        """Load one scorer while the lifecycle lock is held."""
        spec = self.specs[name]
        if spec.backend != "native" or name in self._scorers:
            return
        scorer_cls = _load_native_scorer(spec.native["scorer"])
        kwargs = dict(spec.native.get("kwargs", {}))
        if spec.model_path is not None:
            kwargs.setdefault("model_path", spec.model_path)
        # Ray normally rewrites the visible-device environment for an actor.
        # Its accelerator ID is therefore a physical allocation ID, whereas
        # torch needs the process-local current index (often ``0``). Use
        # verl's platform helper rather than feeding the physical ID to
        # ``torch.device``.
        kwargs.setdefault("device", torch.device(get_device_name(), get_device_id()))
        self._scorers[name] = scorer_cls(**kwargs)

    async def score(self, name: str, prompt: str, image) -> dict:
        async with self._lock:
            self._inflight += 1
            self._idle.clear()
            try:
                await self._wake_up_locked(name)
            except BaseException:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()
                raise
        try:
            async with self._score_lock:
                scorer = self._scorers[name]
                pil_image = _to_pil_image(image)
                score_fn = getattr(scorer, "score", None)
                if score_fn is None:
                    result = await asyncio.get_running_loop().run_in_executor(None, scorer, prompt, pil_image)
                else:
                    result = await asyncio.get_running_loop().run_in_executor(None, score_fn, [prompt], [pil_image])
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, torch.Tensor):
                result = result.tolist()
            if isinstance(result, (list, tuple)):
                score = float(result[0])
            else:
                score = float(result)
            return {"score": score, "pickscore_raw": score}
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()

    async def sleep(self, name: str | None = None) -> None:
        while True:
            await self._idle.wait()
            async with self._lock:
                # A new score can begin after ``wait`` returns. Holding the
                # same lock that increments ``_inflight`` makes this recheck
                # and scorer removal one atomic lifecycle transition.
                if self._inflight:
                    continue
                names = [name] if name is not None else list(self._scorers)
                for scorer_name in names:
                    scorer = self._scorers.pop(scorer_name, None)
                    if scorer is None:
                        continue
                    close = getattr(scorer, "close", None)
                    if close is not None:
                        result = await asyncio.get_running_loop().run_in_executor(None, close)
                        if inspect.isawaitable(result):
                            await result
                gc.collect()
                _empty_accelerator_cache()
                return


def build_native_executor(specs: dict[str, RewardDeploymentSpec]) -> NativeRewardExecutor:
    """Factory kept separate to make worker construction easy to test."""
    return NativeRewardExecutor(specs)
