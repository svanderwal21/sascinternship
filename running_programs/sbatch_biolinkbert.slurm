#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00
#SBATCH --mem 16G
#SBATCH --output=biolink_%j.txt

module purge
module add library/cuda/11.6.1/gcc.8.3.1

nvcc --version
nvidia-smi
hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

sh /exports/sascstudent/svanderwal2/programs/train_programs/monitor.sh &

python3 /exports/sascstudent/svanderwal2/programs/running_programs/biolink/biolinkbert_run_fromhugging_30-11.py > biolink/biolink.txt

