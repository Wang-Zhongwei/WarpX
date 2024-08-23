#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --job-name=mpi_vs_openmp
#SBATCH --account=pas0035
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --array=0-14

# Define combinations of ntasks-per-node and cpus-per-task
declare -a NTASKS_PER_NODE=(1 2 4 8)
declare -a CPUS_PER_TASK=(1 2 4 8)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$((${#NTASKS_PER_NODE[@]} * ${#CPUS_PER_TASK[@]}))

# Ensure the job array size matches the total number of combinations
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_COMBINATIONS} ]; then
    echo "Error: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) exceeds total combinations (${TOTAL_COMBINATIONS})"
    exit 1
fi

# Calculate the index for the current job array task
NTASKS_INDEX=$((SLURM_ARRAY_TASK_ID / ${#CPUS_PER_TASK[@]}))
CPUS_INDEX=$((SLURM_ARRAY_TASK_ID % ${#CPUS_PER_TASK[@]}))

NTASKS=${NTASKS_PER_NODE[$NTASKS_INDEX]}
CPUS=${CPUS_PER_TASK[$CPUS_INDEX]}

source ${HOME}/pitzer_cpu_warpx.profile
export OMP_NUM_THREADS=${CPUS}

# executable & inputs file or python interpreter & PICMI script here
EXE=${HOME}/src/warpx/build/bin/warpx.2d
INPUTS=inputs_2d_mpi_scan

srun --ntasks=${NTASKS} --cpus-per-task=${CPUS} ${EXE} ${INPUTS} >./logs/${SLURM_JOB_NAME}_${NTASKS}_${CPUS}_${SLURM_JOBID}.log 2>&1
