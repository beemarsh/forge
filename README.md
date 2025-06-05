Download miniconda and install in directory where we store all our required builds:

We are not using miniforge3 module.

More here: https://www.anaconda.com/docs/getting-started/miniconda/install#linux

 

```bash

cd /to_my_environment_directory

mkdir ./miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh

bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3

rm ./miniconda3/miniconda.sh

```

> Note: Make sure that conda didn't create any initialziations scripts in .bashrc or .zshrc

 

Now we manually initialize conda and create an environemnt

```bash

source ./miniconda3/etc/profile.d/conda.sh

conda create -p ./forge-conda-env python=3.12.4

```

In module dir, you have a module that will load necessary modules and activate the conda environemnt. Check the file and point to the correct environment file.

```bash

module use ./modules

module load forge

```

 

Now the conda environment is activated. Make sure the environment is activated and python is pointing to the right env.

```bash

which python

```

The output should point to the python in conda env instead of system default python.

 

Now install pytorch wheel:

```bash

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1

```

 

Install other python packages:

```bash

pip install -r requirements.txt

```

 

Now build RCCL:

We will use `enhanced_dump` branch commit `3dbc8009432a484e60fb5026da519e9310a6c225` from this repo: https://github.com/corey-derochie-amd/rccl/tree/enhanced_dump

 

```bash

git clone --recursive -b enhanced_dump https://github.com/corey-derochie-amd/rccl rccl_repo

cd rccl_repo

git checkout 3dbc8009432a484e60fb5026da519e9310a6c225

mkdir build

cd build

cmake -DCMAKE_INSTALL_PREFIX=/your--dir/forge_env/rccl -DCMAKE_BUILD_TYPE=Release ..

make -j 16

make install

```

This will install the libraries in rccl directory.

 

 

 

Now install aws-ofi-rccl

```bash

mkdir aws-ofi-rccl

PLUG_PREFIX=$PWD/aws-ofi-rccl

git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl aws-rccl-repo

cd aws-rccl-repo

./autogen.sh

CC=hipcc CFLAGS=-I/opt/rocm-6.1.3/include ./configure \

    --with-libfabric=/opt/cray/libfabric/1.22.0 \

    --with-rccl=/opt/rocm-6.1.3 \

    --with-hip=/opt/rocm-6.1.3 \

    --with-mpi=$MPICH_DIR \

    --prefix=$PLUG_PREFIX

make

make install

```

 

Now install mpi4py

```bash

MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

```

 

Now install flash-attn from this repo: https://github.com/ROCm/flash-attention

Make sure `triton==3.0.0` is installed

```bash

git clone --recursive https://github.com/ROCm/flash-attention

cd flash-attention

git checkout v2.6.3-cktile

MAX_JOBS=4 FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install

```

 

Install pytorch-lightning

```

pip install pytorch-lightning==2.5.0.post0

```

pip install adan-pytorch==0.1.0

 

Build Mamba:

```bash

git  clone  https://github.com/state-spaces/mamba.git

git checkout v2.2.2

cd  mamba

MAX_JOBS=4 pip  install  --no-build-isolation  .

```

 

Install AMDSMI

```bash

cp -r $ROCM_PATH/share/amd_smi ~/amdsmi_source

cd ~/amdsmi_source

pip install .

```

 

Install APEX

```bash

git clone --recursive https://github.com/ROCm/apex -b release/1.3.0

cd apex

MAX_JOBS=4 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

```

 

Install exodusii

```bash

git clone --recursive https://github.com/sandialabs/exodusii

cd exodusii

python -m pip install .

```

Install

```bash

conda install conda-forge::menuinst=2.1.2

```

 

Update deepspeed to 0.14.4

 

 

 

 

AGAIN:::

 

```bash

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4

pip install triton==3.2.0

git clone --recursive https://github.com/Dao-AILab/flash-attention

cd flash-attention

git checkout v2.6.3

git submodule init

git submodule update --init --recursive

 

DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_CCL_COMM=1 DS_BUILD_TRANSFORMER=1 pip install --no-binary=deepspeed deepspeed

```

 

 

So everytime, I try i get a hang:

This issue tells that it might be torch and deepspeed versions

 

https://github.com/huggingface/trl/issues/2377

 

So I will downgrade pytorch

 

 

WIth VENV:

 

```bash

python -m venv forge_env

MPICC="mpicc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

pip install packaging ninja wheel

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1

 

git clone --recursive https://github.com/Dao-AILab/flash-attention

cd flash-attention

git checkout v2.6.3

git submodule init

git submodule update --init --recursive

pip install triton==3.3.0

MAX_JOBS=120 FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install

 

git clone --recursive https://github.com/EleutherAI/DeeperSpeed -b v3.0

cd DeeperSpeed

MAX_JOBS=120 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_CCL_COMM=1 DS_BUILD_TRANSFORMER=1 pip install .

 

pip install -r requirements.txt

 

MAX_JOBS=64 python megatron/fused_kernels/setup.py install

```

 

WITH apptainer:

```
