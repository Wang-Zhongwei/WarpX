# please set your project account
export proj="" # change me! 

# remembers the location of this script
export MY_H100_PROFILE=$(cd $(dirname $BASH_SOURCE) && pwd)"/"$(basename $BASH_SOURCE)
if [ -z ${proj-} ]; then
  echo "WARNING: The 'proj' variable is not yet set in your $MY_H100_PROFILE file! Please edit its line 2 to continue!"
  return
fi

export SW_DIR="${HOME}/sw/osc/cardinal/h100"

module purge
module load cmake/3.25.2
module load intel/2021.10.0
module load cuda/12.4.1
module load openmpi-cuda/5.0.2

# optional: for python binding support
module load miniconda3/24.1.2-py310
export VENV_NAME="warpx-cardinal-h100"
if [ -d "${SW_DIR}/venvs/${VENV_NAME}" ]; then
  source ${SW_DIR}/venvs/${VENV_NAME}/bin/activate
fi

# an alias to request an interactive batch node for one hour
#   for parallel execution, start on the batch node: srun <command>
alias getNode="salloc -N 1 --ntasks-per-node=2 --cpus-per-task=20 --gpus-per-task=h100:1 -t 1:00:00 -A $proj"
# an alias to run a command on a batch node for up to 30min
#   usage: runNode <command>
alias runNode="srun -N 1 --ntasks-per-node=2 --cpus-per-task=20 --gpus-per-task=h100:1 -t 1:00:00 -A $proj"

# optional: for PSATD in RZ geometry support
export CMAKE_PREFIX_PATH=${SW_DIR}/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=${SW_DIR}/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=${SW_DIR}/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${SW_DIR}/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH

# optional: for QED lookup table generation support
# use self-installed boost
export CMAKE_PREFIX_PATH=${SW_DIR}/boost-1.82.0:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=${SW_DIR}/boost-1.82.0/lib:$LD_LIBRARY_PATH

# optional: for openPMD support (hdf5 and adios2)
export PATH=$SW_DIR/hdf5-1.14.4-3/bin:$PATH
export LD_LIBRARY_PATH=$SW_DIR/hdf5-1.14.4-3/lib:$LD_LIBRARY_PATH

export CMAKE_PREFIX_PATH=${SW_DIR}/c-blosc-1.21.6:$CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=${SW_DIR}/adios2-2.10.1:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=${SW_DIR}/c-blosc-1.21.6/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${SW_DIR}/adios2-2.10.1/lib64:$LD_LIBRARY_PATH
export PATH=${SW_DIR}/adios2-2.10.1/bin:${PATH}

# avoid relocation truncation error which result from large executable size
export CUDAFLAGS="--host-linker-script=use-lcs" # https://github.com/ECP-WarpX/WarpX/pull/3673
export AMREX_CUDA_ARCH=9.0 # 7.0: h100, 8.0: h100, 9.0: H100 https://github.com/ECP-WarpX/WarpX/issues/3214

# compiler environment hints
export CC=$(which gcc)
export CXX=$(which g++)
export FC=$(which gfortran)
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=${CXX}