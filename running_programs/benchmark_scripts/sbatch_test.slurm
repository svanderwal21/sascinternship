#!/bin/bash
#
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem 32G
#SBATCH --output gpt_%j.txt
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=samvdwal@outlook.com

module purge

python3 /exports/sascstudent/svanderwal2/programs/running_programs/benchmark_scripts/test_data_run.py


