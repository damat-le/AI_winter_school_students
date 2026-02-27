#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=tra26_minwinsc
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --mem=450GB
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=checkpoint_convert
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output /leonardo_work/tra26_minwinsc/LLM_challenge_2/logs/%x.%j.out
#SBATCH --error /leonardo_work/tra26_minwinsc/LLM_challenge_2/logs/%x.%j.err
#SBATCH --exclude=lrdn[1261-3456]

source /leonardo_work/tra26_minwinsc/pyenvs/LLM_Challenge_train/bin/activate

python ./torchtitan/scripts/checkpoint_conversion/convert_to_hf.py \
    ./outputs/checkpoint/step-100 ./outputs/checkpoint/step-1100/converted \
    --hf_assets_path ./data/huggingface/Qwen3-1.7B --model_name qwen3 --model_flavor 1.7B 