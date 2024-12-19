#!/bin/bash
#SBATCH --time=0:30:0
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --job-name=compile
#SBATCH --account=PAS0035
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

cd $HOME/src/warpx
rm -rf build_debug
source $HOME/pitzer_cpu_warpx.profile

cmake -S . -B build_debug -DWarpX_DIMS="2" -DCMAKE_BUILD_TYPE=Debug -DAMReX_ASSERTIONS=ON -DAMReX_BOUND_CHECK=ON
cmake --build build_debug -j 48