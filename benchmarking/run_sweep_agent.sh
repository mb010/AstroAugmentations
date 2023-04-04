#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --nodes=1
#SBATCH --constraints=A100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=AA_GMNIST
#SBATCH --output=./logs/%x.%j.out
#SBATCH --error=./logs/%x.%j.err
#SBATCH --exclude=compute-0-8
#SBATCH --array=0-1%2

sleep ${SLURM_ARRAY_TASK_ID}s
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SWEEP_ID='mb010/GalaxyMNIST_galahad/tjunjznz'
source /
wandb agent $SWEEP_ID
