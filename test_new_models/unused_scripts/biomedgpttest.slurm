#!/bin/bash
#
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:59:00
#SBATCH --mem 48G

module purge
#module add library/cuda/11.6.1/gcc.8.3.1

nvcc --version
nvidia-smi
hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

#sh /exports/sascstudent/svanderwal2/programs/train_programs/monitor.sh &

python3 biomedtest.py
