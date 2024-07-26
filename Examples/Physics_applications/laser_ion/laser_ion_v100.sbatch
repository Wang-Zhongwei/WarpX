#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --job-name=laser_ion_v100
#SBATCH --account=pas0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# Pitzer cluster has 32 GPU nodes with dual Intel Xeon 6148 and dual V100 (16GB) GPUs and 42 nodes with dual Intel Xeon 8268 and dual V100 (32GB) GPUs. https://www.osc.edu/resources/technical_support/supercomputers/pitzer

source ${HOME}/pitzer_v100_warpx.profile
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "GPU Information:"
nvidia-smi

# pass this option to warpx executable only if compiled with mvapich2-gdr or openmpi-cuda
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# executable & inputs file or python interpreter & PICMI script here
EXE=${HOME}/src/warpx/build_v100/bin/warpx.2d
INPUTS=inputs_2d

srun --cpu-bind=cores ${EXE} ${INPUTS} ${GPU_AWARE_MPI} >./logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.log 2>&1
