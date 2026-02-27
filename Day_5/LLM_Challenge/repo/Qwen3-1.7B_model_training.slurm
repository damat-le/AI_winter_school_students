#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=tra26_minwinsc
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --mem=450GB
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=qwen3_1.7b_model_train
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output /leonardo_work/tra26_minwinsc/LLM_challenge_2/logs/%x.%j.out
#SBATCH --error /leonardo_work/tra26_minwinsc/LLM_challenge_2/logs/%x.%j.err
#SBATCH --exclude=lrdn[1261-3456]

source /leonardo_work/tra26_minwinsc/pyenvs/LLM_Challenge_train/bin/activate

export HF_DATASETS_OFFLINE=1  # Force offline mode
export DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"
# Ensure that P2P is available
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

export PYTHONPATH=./

CONFIG_FILE=${CONFIG_FILE:-"./train_model_configs/qwen3_1.7b.toml"}

dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
# Replace the srun line with:
srun torchrun \
    --nnodes 4 \
    --nproc-per-node 4 \
    --rdzv-backend c10d \
    --rdzv-endpoint $head_node_ip:29500 \
    --rdzv-id $SLURM_JOB_ID \
    -m torchtitan.train \
    --job.config-file $CONFIG_FILE \
    --model.tokenizer-path data/huggingface/Qwen3-1.7B \
    --training.dataset synth

dcgmi profile --resume

