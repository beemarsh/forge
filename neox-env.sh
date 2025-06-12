module load PrgEnv-gnu
module load gcc/10.3.0
module load rocm/5.3.0
module load miniforge3


export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a

export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/hip/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lustre/orion/gen243/proj-shared/bbdir/forge/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

source activate /lustre/orion/gen243/proj-shared/bbdir/forge/conda-env-forge
#conda config --append envs_dirs /lustre/orion/gen243/proj-shared/bbdir/forge/conda-env-forge
#source activate conda-env-forge
