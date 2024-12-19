#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --job-name=laser_ion_cpu
#SBATCH --account=pas0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

source ${HOME}/cardinal_cpu_warpx.profile
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# executable & inputs file or python interpreter & PICMI script here
EXE=${HOME}/src/warpx/build_cardinal_cpu/bin/warpx.2d
INPUTS=inputs_2d

srun --cpu-bind=cores ${EXE} ${INPUTS} >./logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.log 2>&1
