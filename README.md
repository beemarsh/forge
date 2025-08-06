## 1. Environment Setup
Environment will be located in a directory say: /lustre/orion/..../{env_dir}/
It has two directories: **py312** and **build**.

### Step 1: Initial Environment Setup

First, load the required programming environment and ROCm modules.

```bash
module  load  PrgEnv-gnu/8.6.0
module  load  miniforge3/23.11.0-0
module  load  rocm/6.4.1
module  load  craype-accel-amd-gfx90a
export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
export CXX=$(which g++)
```

### Step 2: Create and Activate Conda Environment

Make sure to replace the the directory in following code. It's recommended to install it in a shared lustre directory (`/lustre/orion/`) to ensure sufficient storage space.
```bash
# Create the env
conda create -p /lustre/orion/{..env_dir..}/py312 python=3.12
# Activate the env 
source activate /lustre/orion/{..your_dir..}/py312
```


### Step 3: Install Core Dependencies

1.  **Install pytorch:**

    ```bash
    pip install ninja
    pip  install  torch==2.8.0  torchvision==0.23.0  torchaudio==2.8.0  --index-url  https://download.pytorch.org/whl/rocm6.4
    ```

2.  **Install mpi4py:**
    ```bash
    MPICC="cc -shared"  pip  install  --no-cache-dir  --no-binary=mpi4py  mpi4py
    ```
3.  **Install deepspeed (DeeperSpeed fork v3.0):**
    ```bash
    cd build
    git clone https://github.com/EleutherAI/DeeperSpeed -b v3.0
    cd DeeperSpeed
    DS_BUILD_FUSED_LAMB=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1  DS_BUILD_UTILS=1 pip install .
    ```
4.  **Install other packages:** 
    ```bash
    pip install -r requirements.txt
     ```

> ⚠️  Make sure the python and pip are from conda environment.
> To check, type `which python` or `which pip` in the terminal.
> This should point to your conda environment  and not to /usr/bin/python.

### Step 4: Build and Configure AWS OFI RCCL Plugin

This plugin is necessary for efficient communication at scale.

1.  **Clone the repository and run autogen:**
    ```bash
    mkdir build
    cd build
    
    rocm_version=6.4.1
    libfabric_path=/opt/cray/libfabric/1.22.0
    cd /path/to/your/workspace
    git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
    cd aws-ofi-rccl
	./autogen.sh
	
	export  LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
	PLUG_PREFIX=$PWD

	CC=hipcc  CFLAGS=-I/opt/rocm-$rocm_version/include  ./configure  \
	--with-libfabric=$libfabric_path  --with-rccl=/opt/rocm-$rocm_version  --enable-trace  	\
	--prefix=$PLUG_PREFIX  --with-hip=/opt/rocm-$rocm_version  --with-	mpi=$MPICH_DIR

	make
	make  install
    ```


3.  **Add the plugin to your environment:**

    Export the path to the newly built library.

    ```bash
    export LD_LIBRARY_PATH=$PLUG_PREFIX/lib:$LD_LIBRARY_PATH
    ```
> ⚠️  Make sure the to match the LD_LIBRARY_PATH in `job.sb` as well.


### Step 5: Fused Kernels:
This will compile the fused kernels.
Make sure you have `export CXX=/opt/cray/pe/gcc-native/14/bin/g++`.
 Run the following command from the `forge` directory.
```python
python
from megatron.fused_kernels import load
load()
```
>⚠️ Important
If it fails, before recompiling, delete all the files under `megatron/fused_kernels/build` and also delete all the hip files.

### Step 6: Dataset

1.  **Create Data Path:** Ensure the directory for the tokenized dataset exists, as specified in the config file.
2. **Download Dataset:** Download tokenized data from: https://doi.ccs.ornl.gov/ui/doi/453. Place both `bin` and `idx` file as well as the `vocab` file as configured in config file.

### Step 7: Modify module file
Modify the module file `forge.lua` inside `forge/modules`. Make sure that environment location is correct.

### Step 8: Modify job script
Modify the job script so that the values are correct, specially `LD_LIBRARY_PATH` and loading of module `forge.lua`.

## 2. Test Design

### 2.1. Test Setup

-   The **job script** is responsible for loading the required **modules** and **Conda environment** using the provided _modulefile_.
    
-   The script **unsets the `PYTHONPATH`** set by the test harness.
    
    > **Note:** This step is a temporary workaround and will be fixed in future updates. (Issues with filelock)


### 2.2. Directory and Configuration Setup

-   The script **creates necessary log directories** for storing outputs and logs.
    
-   Environment variables required for the training run are set accordingly.
    
-   A **host file** is generated to specify node allocation and communication details.
    
    -   The path to this host file is then **referenced in the test configuration file**.


### 2.3. Training Execution

-   The training job is launched using the prepared **configuration files**.
    
-   During execution:
    
    -   Training progress and performance data are logged to **TensorBoard logs** and a deepspeed profiler logs performance metrics to **flops.log**.
        
    -   All relevant outputs are stored in the designated log directories.

