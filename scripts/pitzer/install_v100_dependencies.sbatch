#!/bin/bash
#SBATCH --time=1:00:0
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --job-name=install_v100_dependencies
#SBATCH --account=PAS0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

cd $HOME/src/warpx
source $HOME/pitzer_v100_warpx.profile
./Tools/machines/pitzer-osc/install_v100_dependencies.sh > ./logs/install_v100_dependencies_${SLURM_JOB_ID}.log 2>&1