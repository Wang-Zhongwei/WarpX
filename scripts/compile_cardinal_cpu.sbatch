#!/bin/bash
#SBATCH --time=0:30:0
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --job-name=compile_cardinal_cpu
#SBATCH --account=PAS0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

cd $HOME/src/warpx
rm -rf build_cardinal_cpu
source $HOME/cardinal_cpu_warpx.profile

cmake -S . -B build_cardinal_cpu -DWarpX_DIMS="1;2;3"
cmake --build build_cardinal_cpu -j 48