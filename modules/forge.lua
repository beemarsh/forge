whatis("FORGE PY312 TORCH2.8.0")
whatis("Description: A python anaconda environment for FORGE model training (OLCF-6 Benchmark).")

conflict('cray-python')
load('PrgEnv-gnu/8.6.0')
load('miniforge3/23.11.0-0')
load('rocm/6.4.1')
load('craype-accel-amd-gfx90a')

setenv("HCC_AMDGPU_TARGET", "gfx90a")
setenv("PYTORCH_ROCM_ARCH", "gfx90a")
setenv("CXX", "/opt/cray/pe/gcc-native/14/bin/g++")


local prefix = "/lustre/orion/gen243/world-shared/bbhusal/FORGE_TRAINING/TORCH2.8.0_ROCM6.4.1"

local conda_env_path = "" .. prefix .. "/env/py3.12"

execute {cmd="source activate " .. conda_env_path .. "",modeA={"load"}}

execute {cmd="conda deactivate" ,modeA={"unload"}}
