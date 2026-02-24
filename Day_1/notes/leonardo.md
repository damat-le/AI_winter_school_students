# Leonardo Supercomputer

Leonardo supercomputer: 
* ~6MW per day (like a small town)
* Still in the top10 supercomputers in the world

* 5 Login Nodes
* Compute nodes:
    * Booster partition: has GPUs, Peak performance: 306 PFlops
    * Data centric partition: No GPUs, Peak performance: 13 PFlops
* Topology: dragonfly+
* Workload manager: SLURM
* Storage: 
    * Fast tier: SSD, Capacity 5.4 PB, Speed 1.4 TB/s
    * Capacity tier: HDD, Capacity 106 PB, read speed 744 GB/s, write speed 620 GB/s

## How to access:

* register on the userdb portal (https://userdb.hpc.cineca.it/) 
* get associated with a project (for example from ISCRA calls)
* request access to the system
* You will receive two emails (one is a 2FA setup)

Alternatively: test access (small duration, no project association needed): superc@cineca.it

### Possible projects:
* ISCRA calls
* EuroHPC calls

### How to access:

first time:
* activate 2FA
* install smallstep client

Anytime you want to access, you need to generate ssh certificate (12h validity) and use it to ssh into the system.

You can enter the system using a graphical interface by appending -X to the ssh command.

## Filesystem

Home directory: 
* 50 GB per user
* user specific (persistent across projects)
* not ment to store output results of jobs
* based on HDD

Public directory:
* 50 GB per user
* shared among all users

Scratch directory:
* in principle no storage limit, but there is a policy that deletes files that are older than 40 days
* no backup
* useful for temporary large files, for example output of jobs
* then you have to move important data somewhere else
* based on HDD

Work directory:
* acoount specific (project)
* 1 TB per account 
* shared among all users of the same project
* permanent untill the project is active

Fast directory:
* similar to WORK dir
* fast I/O
* only on leonardo

Tmp directory:
* local to each node
* job specific

Dres directory:
* long storage on demand
* shared among accounts and platforms (not leaonardo)

All filesystems are based on Lustre.
To check your areas, disk usage and quotas: `cindata`.


## Software

Everything is managed by modules. 

Modules are divided into categories: compilers, libraries, applications, tools, data, etc.

Each category is divided into profiles: base / advanced / domain specific (astrophysics, deep learning, etc.).

Available modules: `module av`

Modules are managed/installed with Spack.

To load a specific profile: `module load profile/profile-name`. This will add that profile to the ones already loaded.

`module show ...` 
`modmap -m module-name` to check the dependencies of a module.

some module are compiled for GPU, some for CPU only. When you see intel in the module name / version than probably is CPU only.

To install different softwares:
* install without sudo permissions (cmake etc.)
* with pip in virtual env
* with spack

## Programming environments

Compilers: 
* gcc, nvhpc, intel oneAPI, cuda

MPI libraries:
* openmpi, intel oneAPI mpi 


## Production environment

* Login nodes are not meant to run jobs, they have 10 minute time limit. No GPUs. 

* Compute nodes are meant to run jobs. Two ways to run jobs:
    * batch mode: using `sbatch` to specify a job script with the resources needed and the commands to execute. The job is then cued and scheduled.
    * interactive mode: using `srun` (the session starts directly on the compute node) or `salloc` (a new session is started on the login node, then you can use `srun` to start a session on the compute node). 
    By typing `exit` you exit the interactive job.

* Compute nodes are shared among users, but resources are exclusive.


## Account for this winter school: `tra26_minwinsc` on booster partition (`boost_usr_prod`)

* `sbatch --account=tra26_minwinsc --partition=boost_usr_prod job_script.sh` to submit a job
* `saldo -b` or `saldo -b --dcgp` to check the balance of the account

On the datacentric partition it is possible to request more space for a node: `#SBATCH --gres=tmpfs:100GB` (for example) to have 100 GB of temporary storage on the node. This is useful for data intensive jobs.

Nodes are not connected to the internet. How to have data on compute nodes?

On each booster node:
* 32 cores
* 4 gpus
* 494000 MB of RAM

On each datacentric node:
* 112 cores
* no gpus
* 494000 MB of RAM
* 3TB of local storage in the temporary directory ($TMPDIR)