
# FORGE: OLCF-6 Benchmark

**Original Paper:** https://doi.org/10.1145/3581784.3613215

**Author's Github:** https://github.com/at-aaims/forge

FORGE is a suite of open-source foundation models developed to advance scientific research and discovery. These models, with up to 26 billion parameters, were trained on a specialized corpus of over 200 million scientific articles to excel at domain-specific tasks.

This project documents the integration of FORGE into a test harness as a part of the OLCF-6 benchmark suite. It is designed to stress-test system capabilities for demanding, large-scale scientific AI workloads. The benchmark specifically targets performance in low-precision compute, communication bandwidth, and overall energy efficiency. It includes three model sizes (FORGE-S/M/L) and specific evaluation metrics for a comprehensive assessment. This allows for rigorous evaluation of system performance under realistic LLM training conditions.

**Link to Forge Documentation:** https://www.olcf.ornl.gov/wp-content/uploads/OLCF-6_FORGE_description.pdf

## 1. Reproducing the Environment

This guide provides the steps to set up the necessary environment to run the FORGE benchmark on a system like Frontier.

### Step 1: Initial Environment Setup

First, load the required programming environment and ROCm modules. These exports are crucial for targeting the correct GPU architecture.

**Note:** If you exit your shell session, you must reload these modules and re-export the variables upon returning.

```bash
module load PrgEnv-gnu
module load gcc/10.3.0
module load rocm/5.3.0
#module load miniforge3

export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
```

### Step 2: Create and Activate Conda Environment

First install miniconda. You can use miniforge by doing: ```module load miniforge3``` 
However, I prefer to install miniconda3 in my own directory. Follow the steps:
```bash
# Set the desired installation path
INSTALL_DIR="/lustre/orion/..../miniconda3"

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Run the installer silently and specify the installation directory
bash miniconda.sh -b -p "$INSTALL_DIR"

# Initialize the shell (e.g., bash)
"$INSTALL_DIR/bin/conda" init zsh

source ~/.zshrc  # or ~/.bash_profile depending on system
```

Create a new Conda environment. It's recommended to install it in a shared lustre directory (`/lustre/orion/`) to ensure sufficient storage space.

1.  **Create the environment** (replace `forge_env` with your desired environment name):
    
    ```bash
    conda create -n forge_env python=3.8
    ```
    
2.  **Activate the environment:**
    
    ```bash
    conda activate forge_env
    ```
    

### Step 3: Install Core Dependencies

Install `mpi4py` and `PyTorch` with ROCm 5.3 support.

1.  **Install mpi4py:**
    
    ```bash
    MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
    ```
    
2.  **Install Python Packages:**
    
    ```bash
    pip install -r ./requirements.txt    
    ```
    If installing the above packages dont work, clone the [GPT-NEOX](https://github.com/EleutherAI/gpt-neox) repo and install dependencies as described below: 

	1.  **Clone and check out the correct commit:**
    
    ```bash
    git clone https://github.com/EleutherAI/gpt-neox.git
    cd gpt-neox
    git checkout e48b0c45    
    ```
    
	2.  **Install Python packages:**
    
    ```bash
    pip install -r requirements/requirements.txt
    pip install -r requirements/requirements-tensorboard.txt
    pip install protobuf==3.20.3 pydantic==1.10.21 deepspeed==0.8.2 best_download    
    ```

> ⚠️  Make sure the python and pip are from conda environment.
> To check, type `which python` or `which pip` in the terminal.
> This should point to your conda environment  and not to /usr/bin/python.

### Step 4: Build and Configure AWS OFI RCCL Plugin

This plugin is necessary for efficient communication at scale.

1.  **Clone the repository and run autogen:**
    ```bash
    cd /path/to/your/workspace
    git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
    cd aws-ofi-rccl
    ./autogen.sh    
    ```
    
2.  **Configure and build the plugin:**
    
    ```bash
    # Set the installation prefix
    PLUG_PREFIX=$PWD
    
    # Configure the build
    CC=hipcc CFLAGS=-I/opt/rocm-5.3.0/rccl/include ./configure \
        --with-libfabric=/opt/cray/libfabric/1.20.1 \
        --with-rccl=/opt/rocm-5.3.0/rccl \
        --with-hip=/opt/rocm-5.3.0/hip \
        --with-mpi=$MPICH_DIR \
        --prefix=$PLUG_PREFIX
    
    # Build the plugin
    make
    make install    
    ```


3.  **Add the plugin to your environment:**
    
    Export the path to the newly built library. It is recommended to add this line to your shell profile (`.bashrc`) for persistence.
    
    ```bash
    export LD_LIBRARY_PATH=$PLUG_PREFIX/lib:$LD_LIBRARY_PATH    
    ```
> ⚠️  Make sure the to match the LD_LIBRARY_PATH in `job.sb` as well.
    
    
### Step 5: Build Fused Kernels:
The model requires custom fused kernels for optimal performance. Run the following command from the `gpt-neox` directory: 
```bash
python ./megatron/fused_kernels/setup.py install    
```

### Step 6: Dataset

1.  **Create Data Path:** Ensure the directory for the tokenized dataset exists, as specified in the config file: `data/tokens/`.
2. **Download Dataset:** Download tokenized data from: https://doi.ccs.ornl.gov/ui/doi/453. Place both `bin` and `idx` file as well as the `vocab` file in the `data/tokens` or the directory configured in config file.


***********************************************************


Documentation on DEEPSPEED:

FLOPS profiler : https://www.deepspeed.ai/tutorials/flops-profiler/
Monitoring Metrics: https://www.deepspeed.ai/tutorials/monitor/
Communication Logging: https://www.deepspeed.ai/tutorials/comms-logging/

### Configuration for MODEL
FORGE Description: https://www.olcf.ornl.gov/wp-content/uploads/OLCF-6_FORGE_description-1.pdf

FORGE Dataset: https://doi.ccs.ornl.gov/ui/doi/453

Download the tokenized data and put it in train/data/tokens or define in the config file:
https://doi.ccs.ornl.gov/ui/doi/453
```json
   "data-path": "data/tokens/all_text_document",
   "vocab-file": "data/tokens/all_vocab.json",
```
#### Model Settings
We can control the model size by tweaking the following parameters and also the batch sizes:
```json
   "num-layers": 24,
   "hidden-size": 2064,
   "num-attention-heads": 24,
   "seq-length": 2048,
   "max-position-embeddings": 2048,
   "norm": "layernorm",
   "pos-emb": "rotary",
   "no-weight-tying": true,
```

```json
   "train_micro_batch_size_per_gpu": 16,
   "gas": 1,
```
#### Pipeline parallelism      
Splitting the layers across different GPUs. More about at deepspeed: https://www.deepspeed.ai/tutorials/pipeline/
By default it is set to 1 in forge-s and diabled in forge-l. It is mentioned as PP in the paper.
```json
   "pipe-parallel-size": 1,
```

#### Model parallelism
Splitting the model across GPUs. It is mentioned as TP (Tensor Parallelism) in the paper
In forge-l and forge-m, the model is split across 2 GPUs. So, 2 GPUs are needed to hold a copy of the model for forge-l. So, if there are 800 GPUs, then there are 400 copies (data parallelism).
For forge-s it is set to 1.
For more details, refere the paper section 4.

For a full node TP=8, a single layer distributed across 8 GCD and 4 NUMA domain, the performance drops.

So a proper way would be TP=2 so that parallelism is across two GCD in a single MI250x. This way the performance impact is less severe.

On a single node, the achievable performance per GCD is: 63 TFLOPS.
But a single GCD can perform about 77TFLOPS for 1.76B parameter model (Forge-s).

For larger model 175B, for TP=2, performance drops because of limited network bandwidths. Best way is TP=16 (across two nodes).
```json
   "model-parallel-size": 1,
```

#### Data parallelism
After pipeline and model (tensor) parallelism, everything else is left for data parallelism.
### Below this is the original README
***********************************************************************************

# Pre-Training 

The training is adapted from [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) (e48b0c45) 

## Software environment 
- gcc 10.3.0
- Python 3.8
- Pytorch 1.14.0
- Deepspeed 0.7.3
- ROCM 5.1-5.4
- libfabric 1.15.2.0 
- aws-rccl-plugin 66b3b31

## Build 
- Setup environment 
```bash
odule load PrgEnv-gnu
module load gcc/10.3.0
module load rocm/5.1.0
export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p miniconda
```
- build PyTorch
```bash
git clone --recursive -b IFU-master-2022-11-22 https://github.com/ROCmSoftwarePlatform/pytorch
python tools/amd_build/build_amd.py
USE_ROCM=1 MAX_JOBS=4 python setup.py bdist_wheel
```
- build DeepSpeed
```bash
git clone https://github.com/microsoft/DeepSpeed
DS_BUILD_FUSED_LAMB=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1  DS_BUILD_UTILS=1 pip install .
```
- build GPT-NeoX
```bash
git clone https://github.com/EleutherAI/gpt-neox.git
pip install -r requirements/requirements.txt
```

## Configuration  
The example configuration for [forge-mat](./configs/forge-mat.yml)
- input data setup 
```json
  "data-path": "data/mat/tokens/mat_text_document",
  "vocab-file": "data/mat/tokens/mat_vocab.json",
  "tokenizer_type": "HFTokenizer"
```
- parallelism setup 
```json
   "pipe-parallel-size": 1,
   "model-parallel-size": 1
```
- model setup
```json
   "num-layers": 24,
   "hidden-size": 2064,
   "num-attention-heads": 24,
   "seq-length": 2048,
   "max-position-embeddings": 2048,
   "norm": "layernorm",
   "pos-emb": "rotary"
```
- optimizer setup
```json
   "optimizer": {
     "type": "Lamb",
     "params": {
       "lr": 0.012,
       "betas": [0.9, 0.999],
       "eps": 1.0e-8,
     }
   },
   "zero_optimization": {
    "stage": 1,
    "allgather_partitions": True,
    "allgather_bucket_size": 500000000,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 500000000,
    "contiguous_gradients": True,
    "cpu_offload": False
  }
```

## Run
The example [job script](./job.sb) and Frontier [configuration](./configs/frontier.yml) can be used to run on Frontier 
```bash
sbatch job.sb
```

## Community benchmark 
The community benchmarks are evaluated by 
```python
python ./deepy.py evaluate.py -d configs ${MODEL}.yml frontier.yml --eval_tasks sciq arc_easy arc_challenge piqa hendrycksTest-college_physics hendrycksTest-college_chemistry hendrycksTest-college_medicine hendrycksTest-college_computer_science hendrycksTest-sociology openbookqa
```
