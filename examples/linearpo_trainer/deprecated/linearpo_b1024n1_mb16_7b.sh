set -x
project_name='linearpo_prompt_last_math'
experiment_name='b1024n1_mb16_7b_norm_adv'
model_path=huggingface.co/Qwen/Qwen2.5-7B-Instruct
train_files='[data/math/train.parquet]'
test_files='[data/math/test.parquet]'

# 权重采样开关 (默认关闭，设为true启用)
ENABLE_WEIGHTED_SAMPLING=false
# 如果启用权重采样，可以调整以下参数
WEIGHT_UPDATE_INTERVAL=50       # 每50步更新一次权重
WEIGHT_UPDATE_BATCH_SIZE=32     # 权重更新时的批次大小

mkdir -p "outputs/$project_name/$experiment_name"
script_path="${BASH_SOURCE[0]}"
if [ -f "$script_path" ]; then
    cp "$script_path" "outputs/$project_name/$experiment_name/$(basename $script_path)"
    echo "Script copied to outputs/$project_name/$experiment_name/$(basename $script_path)"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=spo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    reward_estimator.enable=True \
    reward_estimator.hidden_size=3584 \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.output_hidden_states=True \
    actor_rollout_ref.actor.output_hidden_states_mode='prompt_last' \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.enable_weighted_sampling=$ENABLE_WEIGHTED_SAMPLING \
    data.weight_update_interval=$WEIGHT_UPDATE_INTERVAL \
    data.weight_update_batch_size=$WEIGHT_UPDATE_BATCH_SIZE \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-mean' \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir="outputs/$project_name/$experiment_name/train" \
    trainer.validation_data_dir="outputs/$project_name/$experiment_name/validate" \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_training_steps=300 \
    2>&1 | tee -a "outputs/$project_name/$experiment_name/output.log"
