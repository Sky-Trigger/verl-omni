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
"""Placement helper for reward workers that execute on local accelerators.

This module deliberately owns no reward lifecycle or reward-manager policy.
It just binds workers to bundles in a trainer-selected accelerator pool. Both
native deployments and legacy custom reward functions use this same mechanism.
"""

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name


def build_accelerator_reward_workers(
    config,
    reward_loop_workers_class,
    accelerator_resource_pool,
    reward_router_address=None,
    reward_executor_specs=None,
):
    """Create workers bound one-per-bundle in an existing accelerator pool."""
    if accelerator_resource_pool is None:
        raise ValueError("Accelerator reward workers require an accelerator resource pool")
    if accelerator_resource_pool.max_colocate_count < 2:
        raise ValueError(
            "Accelerator reward workers require resource_pool.max_colocate_count >= 2 when colocated with ActorRollout"
        )
    placement_groups = accelerator_resource_pool.get_placement_groups(device_name=get_device_name())
    bundles = [
        (placement_group, bundle_index)
        for bundle_index in range(max(accelerator_resource_pool.store))
        for placement_group, local_world_size in zip(placement_groups, accelerator_resource_pool.store, strict=True)
        if bundle_index < local_world_size
    ]

    num_workers = config.reward.num_workers
    if num_workers > len(bundles):
        raise ValueError(f"reward.num_workers ({num_workers}) exceeds accelerator pool size ({len(bundles)})")

    accelerator_options = get_platform().ray_resource_options(1 / accelerator_resource_pool.max_colocate_count)
    return [
        reward_loop_workers_class.options(
            **accelerator_options,
            name=f"reward_loop_worker_{worker_index}",
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
            ),
        ).remote(config, reward_router_address, reward_executor_specs or {})
        for worker_index, (placement_group, bundle_index) in enumerate(bundles[:num_workers])
    ]
