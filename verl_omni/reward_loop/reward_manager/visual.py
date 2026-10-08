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
"""Deprecated visual-only wrapper around the unified reward manager."""

from .adapter import sampling_params_from_rollout, validate_visual_response
from .multi import MultiRewardManager

# Preserve private imports used by existing integrations while moving their
# implementation into shared input adapters.
_sampling_params_from_rollout = sampling_params_from_rollout
_validate_visual_response = validate_visual_response


class VisualRewardManager(MultiRewardManager):
    """Compatibility wrapper using only the visual reward adapter."""

    _adapter_mode = "visual"
    _deprecated_alias = True
    _supports_named_reward_models = False
