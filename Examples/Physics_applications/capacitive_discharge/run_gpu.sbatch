#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --job-name=capacitive_discharge
#SBATCH --account=pas0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

source ${HOME}/pitzer_v100_warpx.profile
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "GPU Information:"
nvidia-smi

GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# executable & inputs file or python interpreter & PICMI script here
EXE=/users/PAS2137/wang15032/src/warpx/build_v100/bin/warpx.2d
INPUTS=inputs_2d

srun --cpu-bind=cores ${EXE} ${INPUTS} ${GPU_AWARE_MPI} >./logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.log 2>&1