# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
project_name='linearpo_mlp_ppo_sample_filter'
experiment_name='b1024_mb128_4b'
model_path=huggingface.co/Qwen/Qwen3-4B-Base
train_files='[data/dapo/train.parquet,data/math/train.parquet]'
test_files='[data/aime25/test_16.parquet,data/aime24/test_16.parquet,data/amc23/test_16.parquet,data/math500/test.parquet,data/minerva/test.parquet,data/olympiad/test.parquet]'


# Weighted sampling configuration
ENABLE_WEIGHTED_SAMPLING=true
WEIGHT_UPDATE_INTERVAL=30
WEIGHT_UPDATE_BATCH_SIZE=1024

# Estimated reward filtering configuration
# This will filter out samples with estimated_reward < 0.05 (too hard) or > 0.9 (too easy)
# Only samples with 0.05 <= estimated_reward <= 0.9 will be used for actor updates
# Critic and reward estimator will still use ALL samples
ENABLE_FILTER=true
FILTER_MIN=0.05
FILTER_MAX=0.9


mkdir -p "outputs/$project_name/$experiment_name"
script_path="${BASH_SOURCE[0]}"
if [ -f "$script_path" ]; then
    cp "$script_path" "outputs/$project_name/$experiment_name/$(basename $script_path)"
    echo "Script copied to outputs/$project_name/$experiment_name/$(basename $script_path)"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=spo \
    reward_estimator.enable=True \
    reward_estimator.model.hidden_size=2560 \
    reward_estimator.offload_to_cpu=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.enable_weighted_sampling=$ENABLE_WEIGHTED_SAMPLING \
    data.weight_update_interval=$WEIGHT_UPDATE_INTERVAL \
    data.weight_update_batch_size=$WEIGHT_UPDATE_BATCH_SIZE \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=$ENABLE_FILTER \
    algorithm.filter_groups.metric=estimated_reward \
    algorithm.filter_groups.estimated_reward_min=$FILTER_MIN \
    algorithm.filter_groups.estimated_reward_max=$FILTER_MAX \
    actor_rollout_ref.actor.output_hidden_states=True \
    actor_rollout_ref.actor.output_hidden_states_mode='prompt_last' \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
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