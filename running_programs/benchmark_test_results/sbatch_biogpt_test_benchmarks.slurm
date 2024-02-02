#!/bin/bash
#
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00
#SBATCH --mem 16G
#SBATCH --output gpt_%j.txt

module purge
module add library/cuda/11.6.1/gcc.8.3.1

nvcc --version
nvidia-smi
hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

echo "Script for generating these results: "
cat /exports/sascstudent/svanderwal2/programs/running_programs/biogpt/biogpt_testing_benchmarks.py

sh /exports/sascstudent/svanderwal2/programs/train_programs/monitor.sh &

echo "Results:"
python3 /exports/sascstudent/svanderwal2/programs/running_programs/biogpt/biogpt_testing_benchmarks.py

echo "CPU Statistics:"
cat /exports/sascstudent/svanderwal2/programs/train_programs/logs/cpu_usage.log
echo "GPU Statistics"
cat /exports/sascstudent/svanderwal2/programs/train_programs/logs/gpu_usage.log

