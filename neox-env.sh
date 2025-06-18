#!/bin/bash

module load PrgEnv-gnu
module load gcc/10.3.0
module load rocm/5.3.0
module load miniforge3

env_path=""
for arg in "$@"; do
  if [[ $arg == --env-path=* ]]; then
    env_path="${arg#*=}"
    break
  fi
done

if [ -n "$env_path" ]; then
  source activate "$env_path"
else
  echo "Error: The --env-path=<path> argument is required." >&2
  exit 1
fi
