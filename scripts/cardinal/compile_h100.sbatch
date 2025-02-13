#!/bin/bash
#SBATCH --time=0:45:0
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --job-name=compile_h100
#SBATCH --account=PAS0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err


cd $HOME/src/warpx
rm -rf build_h100
source $HOME/cardinal_h100_warpx.profile

cmake -S . -B build_h100 -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3" -DWarpX_QED_TOOLS=ON
cmake --build build_h100 -j 48