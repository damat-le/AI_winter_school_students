# Profiling AI systems

Lecturers: Alexandros Paliouras, Guillem Rovira Cortiada

## Introduction

### Communication

Itra-node:
    * NVLink cables (in Leonardo 200 GB/s)

Inter-node:
    * Infiniband (in Leonardo ~25 GB/s)

At software level, we have:
    * NCCL (NVIDIA Collective Communications Library): implement multi-gpu and multi-node communication primitives (e.g., reduce, scatter, gather, broadcast etc.).

Notes:
* IN DDP:
    * we use broadcast only at the beginning of the training to synchronize the weights of the model across all the gpus.
    * after N iterations, we use all-reduce to synchronize the gradients across all the gpus.

* In FSDP:
    * during the forward and backward pass, we use all-gather to gather the weights and gradients across all the gpus.
    * after backward pass, we use reduce-scatter to scatter the gradients across all the gpus.


## Nvidia Nsight Systems

Software level:
* kernel: basic function that runs on the gpu, e.g., matrix multiplication, convolution, etc.
* CUDA API: cpu-gpu interface functions, e.g., cudaMalloc, cudaMemcpy, etc. They allow to allocate memory on the gpu, copy data between cpu and gpu, synch execution, etc.
* NVTX: lightweight annotation api to mark events and ranges in the code. It allows to visualize the execution of the code in the Nsight Systems timeline.

Hardware level:
* SM (streaming multiprocessor): basic computational unit of the gpu. It consists of a number of CUDA cores, which are the actual processing units that execute the kernels. The A100 has 108 SMs, each with 64 CUDA cores, for a total of 6912 CUDA cores.
* GPU threads: smallest execution unit that runs on the gpu. Each thread executes a kernel and can access the shared memory of the SM.
* Warp: group of 32 threads that execute the same instruction at the same time.

## Paraver

It a software developed by Barcelona Supercomputing Center to profile parallel applications. It allows to visualize the execution of the code in a timeline, similar to Nsight Systems, but it also allows to visualize the communication between processes and threads.