#!/bin/bash --login

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH -J "split_data"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <=
# ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket
python3 \
    ../../src/split_data.py \
    ../../configs/user_configs/PATH_TO_CONFIG.yaml