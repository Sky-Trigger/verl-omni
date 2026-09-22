Reward Interface
================================

Last updated: |today| (API docstrings are auto-generated).

VeRL-Omni reward pipelines support both rule-based scoring (e.g. JPEG
compressibility) and model-based generative reward models (e.g. OCR via a
vision-language model served behind an OpenAI-compatible router). Reward
computation is dispatched per sample by reward managers. The modality-neutral
:class:`~verl_omni.reward_loop.reward_manager.MultiRewardManager` runs named
reward terms and preserves their per-model outputs while computing the
configured weighted aggregate. Text, visual, and audio input adapters project
rollout outputs into scorer arguments without changing engine execution.
Legacy modality-specific manager names remain as compatibility wrappers and
may be deprecated in a future release. New configurations should use only
``MultiRewardManager``.
The manager plugs into :class:`~verl_omni.reward_loop.reward_loop.OmniRewardLoopManager` — verl's
:class:`~verl.experimental.reward_loop.RewardLoopManager` extended with
profiler control over the reward-model rollout servers.

.. autosummary::
   :nosignatures:

   verl_omni.reward_loop.reward_loop.OmniRewardLoopManager
   verl_omni.reward_loop.reward_manager.MultiRewardManager
   verl_omni.reward_loop.reward_manager.TextRewardAdapter
   verl_omni.reward_loop.reward_manager.VisualRewardAdapter
   verl_omni.reward_loop.reward_manager.AudioRewardAdapter
   verl_omni.utils.reward_score.default_compute_score_image
   verl_omni.utils.reward_score.http_scorer_client.compute_score
   verl_omni.utils.reward_score.audio_http_scorer_client.compute_score
   verl_omni.utils.reward_score.unified_reward.compute_score_unified_reward

Reward Loop Manager
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.reward_loop.reward_loop.OmniRewardLoopManager
   :members: start_profile, stop_profile

Reward Manager
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.reward_loop.reward_manager.MultiRewardManager
   :members: __init__, run_single, assemble_rm_scores

Reward Input Adapters
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.reward_loop.reward_manager.TextRewardAdapter
   :members: matches, adapt

.. autoclass:: verl_omni.reward_loop.reward_manager.VisualRewardAdapter
   :members: matches, adapt

.. autoclass:: verl_omni.reward_loop.reward_manager.AudioRewardAdapter
   :members: matches, adapt

The manager always runs the adapter for a sample's primary modality. An
auxiliary modality is projected only when at least one configured scorer
explicitly declares the corresponding argument, such as ``solution_audio``.
Accepting ``**kwargs`` alone does not request every auxiliary modality. This
keeps video rewards from validating or copying side-channel audio they do not
consume while allowing audiovisual scorers to opt in explicitly.

Default Score Dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: verl_omni.utils.reward_score
   :members: default_compute_score_image

Built-in Reward Scorers
~~~~~~~~~~~~~~~~~~~~~~~~

JPEG Compressibility
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.jpeg_compressibility
   :members: jpeg_compressibility, jpeg_incompressibility, compute_score

GRM-based OCR Reward
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.genrm_ocr
   :members: compute_score_ocr

HTTP Scorer Client
^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.http_scorer_client
   :members: compute_score

Audio HTTP Scorer Client
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.audio_http_scorer_client
   :members: compute_score

UnifiedReward Scorer
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.unified_reward
   :members: compute_score_unified_reward

Reward Utilities
^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.reward_utils
   :members: video_tensor_to_pil_frames, pil_image_to_base64
