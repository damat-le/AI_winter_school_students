#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --account=tra26_minwinsc         # account name 
#SBATCH --partition=boost_usr_prod       # partition name
#SBATCH --time=00:05:00                  # 5 minutes is plenty for a quick test
#SBATCH --nodes=1                        # Request exactly 1 node
#SBATCH --ntasks-per-node=1              # 1 task is enough to run the diagnostic command
#SBATCH --cpus-per-task=1                # Allocate a few CPUs
#SBATCH --gres=gpu:4                     # Allocate 4 GPUs on the node
##SBATCH --mem=494000                    # Memory
#SBATCH --output=logs/gpu_info_%j.out         # Save output 
#SBATCH --error=logs/gpu_info_%j.err          # Save errors 

mkdir -p logs

echo "====================================================="
echo "Job running on node: $(hostname)"
echo "Date: $(date)"
echo "====================================================="

echo -e "\n---> 1. Basic GPU Information (nvidia-smi):"
srun nvidia-smi

echo -e "\n---> 2. GPU NVLink Topology (nvidia-smi topo -m):"
srun nvidia-smi topo -m

echo -e "\n====================================================="
echo "Test completed successfully!"