#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=tra26_minwinsc
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --mem=450GB
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=qwen3_checkpoint_model_eval_test
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output /leonardo_work/tra26_minwinsc/LLM_challenge_2/logs/%x.%j.out
#SBATCH --error /leonardo_work/tra26_minwinsc/LLM_challenge_2/logs/%x.%j.err
#SBATCH --exclude=lrdn[1261-3456]

set -u

source ./.venv_eval/bin/activate

# Set cache directories to use pre-downloaded data
export HF_HOME=/leonardo_scratch/fast/iGen_train/avalent2/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Comment when downloading the datasets
export HF_DATASETS_OFFLINE=1  # Force offline mode
export TRANSFORMERS_OFFLINE=1

lighteval vllm \
    "./eval_model_configs/Qwen3-1.7B-checkpoint_vllm_model_config.yaml" \
    "mmlu_pro|5,gsm8k|5,ifeval"