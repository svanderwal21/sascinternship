#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00
#SBATCH --mem 16G


module purge
module add library/cuda/11.6.1/gcc.8.3.1

nvcc --version
nvidia-smi
hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

#python3 /exports/sascstudent/svanderwal2/programs/train_programs/monitor.py &

python3 /exports/sascstudent/svanderwal2/programs/semantic_analysis/sss.py

