Here I collected a series of scripts/info that can be useful to interact with the leonardo cluster.

## How to run a script on the cluster

To run a script on the cluster, you can use the `sbatch` command. For example, if you have a script called `my_script.sh`, you can run it with the following command:

```bash
sbatch my_script.sh
```

The sbatch command will submit your script to the cluster's job scheduler, which will then execute it according to the specified resources and scheduling policies. Make sure to include any necessary directives in your script (e.g., `#SBATCH` lines) to specify the resources you need for your job.

Example of  `#SBATCH` lines are: 
```bash
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --account=tra26_minwinsc         # account name 
#SBATCH --partition=boost_usr_prod       # partition name
#SBATCH --time=00:05:00                  # time limit (hh:mm:ss)
#SBATCH --nodes=1                        # Request exactly 1 node
#SBATCH --ntasks-per-node=1              # 1 task is enough to run the diagnostic command
#SBATCH --cpus-per-task=1                # Allocate a few CPUs
#SBATCH --gres=gpu:4                     # Allocate 4 GPUs on the node
##SBATCH --mem=494000                    # Memory
#SBATCH --output=logs/gpu_info_%j.out         # Save output 
#SBATCH --error=logs/gpu_info_%j.err          # Save errors 
```

Note: if you do not use sbatch, the script will run on the login node, which is not recommended for long-running or resource-intensive tasks. In fact, the login node is meant for light tasks and there is a time limit of 10 minutes per process. If you run heavy tasks on the login node, you may create problems to other users on the cluster.