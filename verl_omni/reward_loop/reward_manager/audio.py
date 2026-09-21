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
"""Deprecated audio-only wrapper around the unified reward manager."""

from verl_omni.reward_loop.reward_manager.adapter import AudioRewardAdapter, RewardInputAdapter, extract_audio
from verl_omni.reward_loop.reward_manager.multi import MultiRewardManager


class AudioRewardManager(MultiRewardManager):
    """Deprecated compatibility wrapper using only the audio reward adapter."""

    _adapter_mode = "audio"
    _require_custom_reward_function = True
    _supports_named_reward_models = False
    _extract_audio = staticmethod(extract_audio)

    def _create_reward_adapters(self) -> tuple[RewardInputAdapter, ...]:
        return (AudioRewardAdapter(required=True, audio_extractor=self._extract_audio),)
