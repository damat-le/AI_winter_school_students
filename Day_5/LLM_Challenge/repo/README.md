# CINECA LLM challenge

Hello there, welcome to the CINECA LLM Challenge!
The goal of this challenge is to make you experiment with a multi-GPU, multi-node training of real, production-scale LLMs.

You can find some example script for training three models of different sizes (Qwen 1.7B, Qwen 32B and Llama4 17Bx16) on 4 nodes:
```bash
    Qwen3-1.7B_model_training.slurm
    Qwen3-32B_model_training.slurm
    Llama417Bx16e_model_training.slurm
```
Try to scale each one of them to as many nodes as possible! Remember to maximize the overall throughput while doing so.
Feel free to experiment with different training configuration, you can find them in the `train_model_configs` subfolder.

Scale the nodes, then play with the parallellization settings to achieve the best mean MFU.

If you also wish to evaluate the trained models on real-world benchmarks, check out the "Model Eval" section.

## Evaluation Criteria
The winner team will be the one that achieves the best (highest) mean MFU on the three models Qwen 1.7B, Qwen32B, and LLama4-17Bx16e. The mean MFU is reported in the training logs as `mmfu` in percentage points.

In their final presentations, the teams are required to report the following information:
- The MFU in the starting configuration for each of the three models.
- The MFU achieved after tuning the config on the initial number of nodes (4 nodes on Qwen1.7B and Qwen 32B, 8 nodes on Llama4-17x16)
- The MFU achieved after tuning the config on a upscaled training (8 nodes on Qwen1.7B and Qwen 32B, 16 nodes on Llama4-17x16)
- OPTIONAL: the MFU achieved after tuning the config on a downscaled training for Qwen1.7B (1 or 2 nodes).
- OPTIONAL: If you have time, try scaling up on even more nodes for additional points!

The winning team will be the one with better MFU stability (that is, the one with the lowest drop in MFU as the number of nodes increases), averaged over the three models.


## Installation
###  Model Training: TorchTitan
Model training is done through the [TorchTitan library](https://github.com/pytorch/torchtitan).
First, pip install TorchTitan into a virtual environment, using the requirements-train.txt file as a reference. I suggest to install Torchtitan from source:
```bash
    cd torchtitan
    pip install -e .
```
NB: use the torchtitan source provided in this repo, as I needed to make some adjustments to the original torchtitan code. 
NB 2: I had to install the nightly version of pytorch (torch==2.11.0.dev20260202+cu126) to make things work.
After that, you are ready to go! Test the installation by launching one of these training scripts with sbatch:
- `Qwen3-1.7B_model_training.slurm`
- `Qwen3-32B_model_training.slurm`
-  `Llama417Bx16e_model_training.slurm`

(You will need to adjust the paths and some names to point to your local directories.)

The training config are defined in the `train_model_configs` subfolder.

#### Adding a New Dataset
You can use additional datasets rather than the ones already provided by following [this guide](https://deepwiki.com/pytorch/torchtitan/13.2-adding-custom-datasets).

We suggest you to use the [Synth dataset](https://huggingface.co/datasets/PleIAs/SYNTH). Clone it in the `data/huggingface/synth` folder

#### Converting Checkpoints
Before performing evaluations with Lighteval, you need to convert the Pytorch distributed checkpoint to the Huggingface format.
You can do it with the scripts in the `torchtitan/scripts/checkpoint_conversion` folder.
See the `Qwen3-1.7B_checkpoint_converter.slurm` script as an example.

### Model Eval: Lighteval
Model evaluation is done through the [Lighteval](https://github.com/huggingface/lighteval) library.
Create a virtualenv (different from the torchtitan one) and [install lighteval](https://huggingface.co/docs/lighteval/v0.13.0/en/installation) in it.
In particular, install the 'vllm' package extension.
Alternatively, you can use the requirements-eval.txt file included in this folder.

NB: make sure that transformers is at version 4.55.0, it will clash with vllm otherwise.

You are ready to go! Test the installation by launching one eval script with sbatch:
- SmoLM-1.7B_model_eval.slurm
- Qwen3-32B_model_eval.slurm

(You will need to adjust the paths to point to your local directories.)

The training config are defined in the `eval_model_configs` subfolder.

#### Notes on the Eval:

Since the Leonardo compute nodes do not have internet access, you should first run the script on a login node, to make it cache the benchmark tasks. Make sure to comment the following lines in order to enable online mode:

```bash
    # Comment when downloading the datasets
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
```

The script will download the required tasks, and then crash because it will not find a GPU. You can then re-launch the script in offline mode, using the cached tasks, to perform the final eval.

Sometimes, for single-node evals, it is simpler to start an interactive session with srun: 

```bash
srun --partition=boost_usr_prod --nodes=1 --qos <your-qos> --ntasks=1 --exclusive --gres=gpu:4  --time=08:00:00 --account=<your account> --pty /bin/bash
```

And then run the slurm script inside the session using bash:
    ```bash
        bash <path-to-your-script>.slurm 
    ```