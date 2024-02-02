#!/bin/bash

#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:59:00
#SBATCH --mem 64G

module purge
module add library/cuda/11.6.1/gcc.8.3.1

nvcc --version
nvidia-smi
hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

/exports/sascstudent/svanderwal2/programs/test_new_models/monitor_resources.sh &
MONITOR_PID=$!

python3 classify_model_try.py

kill $MONITOR_PID
