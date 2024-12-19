#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=capacitive_discharge
#SBATCH --account=pas0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

source ${HOME}/pitzer_cpu_warpx.profile
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# executable & inputs file or python interpreter & PICMI script here
EXE=${HOME}/src/warpx/build/bin/warpx.2d
INPUTS=inputs_2d

srun --cpu-bind=cores ${EXE} ${INPUTS} >./logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.log 2>&1    