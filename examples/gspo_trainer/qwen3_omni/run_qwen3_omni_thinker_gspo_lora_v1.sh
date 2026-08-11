#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA training with optional OPD distillation.
#
# Normal GSPO:
#   ENABLE_DISTILLATION=false bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh
# OPD distillation (default):
#   bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh

set -x

export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni

# -----------------------------------------------------------------------------
# Model and data
# -----------------------------------------------------------------------------

MODEL_PATH=${MODEL_PATH:-"/mnt/sfs_turbo/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/data/datasets/gsm8k/train.parquet"}
VAL_FILE=${VAL_FILE:-"/mnt/data/datasets/gsm8k/test.parquet"}

# -----------------------------------------------------------------------------
# Default training settings
# -----------------------------------------------------------------------------

TRAINER_NGPUS=16
EXPERIMENT_NAME="qwen3_omni_thinker_lora"

# -----------------------------------------------------------------------------
# Distillation settings
# -----------------------------------------------------------------------------

ENABLE_DISTILLATION=${ENABLE_DISTILLATION:-true}
DISTILLATION_ARGS=()

if [ "${ENABLE_DISTILLATION}" = "true" ]; then
    # 16 NPUs = 12 student + 4 teacher.
    TRAINER_NGPUS=${TRAINER_NGPUS_DISTILL:-12}
    TEACHER_NGPUS=${TEACHER_NGPUS:-4}

    TEACHER_MODEL=${TEACHER_MODEL:-"${MODEL_PATH}"}
    TEACHER_TP=${TEACHER_TP:-2}
    TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.7}

    # prompt 2048 + response 8192 + one token requested by teacher scoring.
    TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-10241}

    DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
    USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
    DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}

    EXPERIMENT_NAME="qwen3_omni_thinker_lora_distill"

    DISTILLATION_ARGS=(
        distillation.enabled=true
        distillation.nnodes=1
        distillation.n_gpus_per_node=${TEACHER_NGPUS}
        distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
        distillation.teacher_models.teacher_model.inference.name=vllm_omni
        distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
        distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
        distillation.teacher_models.teacher_model.inference.max_model_len=${TEACHER_MAX_MODEL_LEN}
        +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.output_mode=ar
        +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe
        distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
        distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
        distillation.distillation_loss.use_task_rewards=true
        distillation.distillation_loss.distillation_loss_coef=${DISTILLATION_LOSS_COEF}
        distillation.distillation_loss.policy_loss_mode=vanilla
        distillation.distillation_loss.clip_ratio_low=0.2
        distillation.distillation_loss.clip_ratio_high=0.28
    )
fi

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="[1,2,8,64,512,1024,2048]" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.reward_model.rollout.enforce_eager=True \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=gspo \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${TRAINER_NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "${DISTILLATION_ARGS[@]}" \
    "$@"
