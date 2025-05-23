#!/bin/bash

# Parameters
#SBATCH --account=csnlp_jobs
#SBATCH --error=/home/vzarzu/Bridging-the-Chains/logs/slurm/%j_0_log.err
#SBATCH --job-name=cil_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/vzarzu/Bridging-the-Chains/logs/slurm/%j_0_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=900
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/vzarzu/Bridging-the-Chains/logs/slurm/%j_%t_log.out --error /home/vzarzu/Bridging-the-Chains/logs/slurm/%j_%t_log.err /home/vzarzu/miniconda3/envs/csnlp/bin/python -u -m submitit.core._submit /home/vzarzu/Bridging-the-Chains/logs/slurm
