# Parallel AI

Lecturer: Domitilla Brandoni


## Data Parallelism

* each gpu hold a copy of the model 
* each gpu process a different batch of data
* after some backpropagation steps, the gradients are sinchronized across the gpus

Pytorch DDP:
* communication: ring all-reduce
* in the course we use HF accelerate, which is a wrapper around DDP.
    * it uses a config file to specify the number of gpus, the batch size, etc.
    * you will need to run your script with `accelerate launch` instead of `python`.
    * number of machines: corresponds to a node.
    * number of processes per machine: corresponds to the number of gpus per node.
    * In SLURM, you have to add info to the script you want to run with accelerate (slide 18)

How to compare a model trained with DDP and a model trained without DDP?
* not trivial.


## FSDP: Fully Sharded Data Parallel
* each gpu holds only a shard of the model, not a full copy.
* communication: all-gather and scatter

* 1 unit: bunch of layers
* 1 shard: a subset of the parameters of the model

* each gpu receive a mini batch 


## MODEL PARALLELISM

Two types for splitting the model:
* horizontal (tensor parallelism): split the model within a layer, each gpu holds a subset of the parameters of the layer. 
    * keep the number of computational units smaller than the number of gpus in a node to benefit from the communication speed of the NVLink.
    * moving data between nodes is much slower than moving data between gpus in the same node, so it's better to keep the communication within the node.

* vertical (pipeline parallelism): split the model layer-wise, each gpu holds a subset of the layers.
    * ...