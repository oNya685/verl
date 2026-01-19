set -x
project_name='linearpo_mlp_token_ppo_sample_filter_epistemic_norm_adv'
experiment_name='a50_f0.2_0.6_b1024_mb128_4b_0119'
model_path=huggingface.co/Qwen/Qwen3-4B-Base
train_files='[data/dapo/train.parquet,data/math/train.parquet]'
test_files='[data/aime25/test_16.parquet,data/aime24/test_16.parquet,data/amc23/test_16.parquet,data/math500/test.parquet,data/minerva/test.parquet,data/olympiad/test.parquet]'

ENABLE_WEIGHTED_SAMPLING=true
WEIGHT_UPDATE_INTERVAL=30
WEIGHT_UPDATE_BATCH_SIZE=1024

BETA_EXPLORATION=1.0
ENABLE_EPISTEMIC=true
LAMBDA_REG=1.0
ALPHA_SCALE=50

ENABLE_FILTER=true
USE_DYNAMIC_FILTERING=true
FILTER_MIN=0.2
FILTER_MAX=0.6

mkdir -p "outputs/$project_name/$experiment_name"
script_path="${BASH_SOURCE[0]}"
if [ -f "$script_path" ]; then
    cp "$script_path" "outputs/$project_name/$experiment_name/$(basename $script_path)"
    echo "Script copied to outputs/$project_name/$experiment_name/$(basename $script_path)"
fi

export VERL_AUTO_PADDING=True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    reward_estimator.enable=True \
    reward_estimator.model.hidden_size=2560 \
    reward_estimator.offload_to_cpu=False \
    reward_estimator.epistemic_uncertainty.enable=$ENABLE_EPISTEMIC \
    reward_estimator.epistemic_uncertainty.lambda_reg=$LAMBDA_REG \
    reward_estimator.epistemic_uncertainty.alpha_scale=$ALPHA_SCALE \
    reward_estimator.epistemic_uncertainty.use_dynamic_filtering=$USE_DYNAMIC_FILTERING \
    data.enable_weighted_sampling=$ENABLE_WEIGHTED_SAMPLING \
    data.weight_update_interval=$WEIGHT_UPDATE_INTERVAL \
    data.weight_update_batch_size=$WEIGHT_UPDATE_BATCH_SIZE \
    data.beta_exploration=$BETA_EXPLORATION \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=$ENABLE_FILTER \
    algorithm.filter_groups.metric=estimated_reward \
    algorithm.filter_groups.estimated_reward_min=$FILTER_MIN \
    algorithm.filter_groups.estimated_reward_max=$FILTER_MAX \
    actor_rollout_ref.actor.output_hidden_states=True \
    actor_rollout_ref.actor.output_hidden_states_mode='full_response' \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=151 \
    trainer.test_freq=20 \
    trainer.total_training_steps=300 \
    2>&1 | tee -a "outputs/$project_name/$experiment_name/output.log"