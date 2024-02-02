#!/bin/bash
#
#SBATCH --partition=gpu,highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem 24G


nvcc --version
nvidia-smi
hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"
python3 huggingface_tutorial_finetuning.py
