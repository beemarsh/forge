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
module load miniforge3

export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
```

### Step 2: Create and Activate Conda Environment

Make sure to replace the the directory in following code. It's recommended to install it in a shared lustre directory (`/lustre/orion/`) to ensure sufficient storage space.
```bash
# Create the env
conda create -p /lustre/orion/{..your_dir..}/forge_env python=3.8
# Activate the env 
source activate /lustre/orion/{..your_dir..}/forge_env
```


### Step 3: Install Core Dependencies

Install `mpi4py` and `PyTorch` with ROCm 5.3 support.

1.  **Install mpi4py:**

    ```bash
    MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
    ```

2.  **Install Pytorch (ROCM wheel):**

    ```bash
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/rocm5.3
    ```
3.  **Install other packages:**

    ```bash
    pip install -r requirements.txt
     ```
    
    If installing the above packages fail, clone the [GPT-NEOX](https://github.com/EleutherAI/gpt-neox) repo and install dependencies as described below:
    

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
    pip install protobuf==3.20.3
    pip install pydantic==1.10.21
    pip install deepspeed==0.8.2
    pip install best_download
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

    Export the path to the newly built library. It is recommended to add this line to your shell profile (`.zshrc`) for persistence.

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



## 2. Model Configuration


### 2.1. Model and Batch Size Settings

The model's size and training batch size can be controlled by adjusting the following parameters in the configuration file.

**Model Architecture:**

```json
{
  "num-layers": 24,
  "hidden-size": 2064,
  "num-attention-heads": 24,
  "seq-length": 2048,
  "max-position-embeddings": 2048,
  "norm": "layernorm",
  "pos-emb": "rotary",
  "no-weight-tying": true
}
```

**Batch Size:**

```json
{
  "train_micro_batch_size_per_gpu": 16,
  "gas": 1
}
```

## 3. Parallelism Strategies

### 3.1. Pipeline Parallelism (PP)

Pipeline parallelism involves splitting the model's layers across different GPUs to process data in a pipeline fashion. You can learn more at the [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/ "null") Pipeline Parallelism [Tutorial](https://www.deepspeed.ai/tutorials/pipeline/ "null").

-   In the paper, this is referred to as **PP**.
    
-   By default, it is enabled (`"pipe-parallel-size": 1`) in `forge-s` and disabled in `forge-l`.
    
The following is the setting for `forge-s` where PP=1.
```json
{
  "pipe-parallel-size": 1
}
```

### 3.2. Model Parallelism / Tensor Parallelism (TP)

Model parallelism (specifically, Tensor Parallelism) splits the model's tensors (e.g., weight matrices) across multiple GPUs. This means a single copy of the model is distributed.

-   In the paper, this is referred to as **TP**.
    
-   For `forge-l` and `forge-m`, the model is split across 2 GPUs (`"model-parallel-size": 2`), requiring 2 GPUs to hold one full copy of the model.
    
-   For `forge-s`, this is set to 1.
    
-   Refer to **Section 4** of the paper for more details.
    

**Performance Considerations:**

-   For a full node with `TP=8` (a single layer distributed across 8 GCDs and 4 NUMA domains), performance drops significantly.
    
-   A more optimal configuration is `TP=2`, keeping the parallelism within a single MI250x, which lessens the performance impact.
    
-   A single GCD on a node can achieve **63 TFLOPS**. However, for a 1.76B parameter model (`Forge-s`), a single GCD can reach **77 TFLOPS**.
    
-   For very large models (e.g., 175B parameters), `TP=2` can be limited by network bandwidth. A better approach is `TP=16`, which spans two nodes.
    
This is the setting for `forge-s` where TP=1.
```json
{
  "model-parallel-size": 1
}
```

### 3.3. Data Parallelism

After accounting for pipeline and model parallelism, the remaining resources are used for data parallelism. For example, with `forge-l` using `TP=2` on a system with 800 GPUs, there will be 400 data-parallel copies of the model.



    


## 1. DeepSpeed Documentation

For more background on the tools and techniques used in this project, please refer to the official DeepSpeed documentation:

-   **FLOPS Profiler:** [https://www.deepspeed.ai/tutorials/flops-profiler/](https://www.deepspeed.ai/tutorials/flops-profiler/ "null")
    
-   **Monitoring Metrics:** [https://www.deepspeed.ai/tutorials/monitor/](https://www.deepspeed.ai/tutorials/monitor/ "null")
    
-   **Communication Logging:** [https://www.deepspeed.ai/tutorials/comms-logging/](https://www.deepspeed.ai/tutorials/comms-logging/ "null")
