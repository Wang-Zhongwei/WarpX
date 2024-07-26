#!/bin/bash
#SBATCH --time=0:30:0
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --job-name=compile
#SBATCH --account=PAS0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

cd $HOME/src/warpx
rm -rf build
source $HOME/pitzer_cpu_warpx.profile

cmake -S . -B build -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
cmake --build build -j 48