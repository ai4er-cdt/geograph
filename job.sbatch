#!/bin/bash
#SBATCH --partition=lotus_gpu
#SBATCH --account=lotus_gpu
#SBATCH --gres=gpu:1 # Request a number of GPUs
#SBATCH --time=12:00:00 # Set a runtime for the job in HH:MM:SS
#SBATCH --mem=128000 # Set the amount of memory for the job in MB.

conda activate gtc-bio2 # Replace with your conda environment
srun python src/train.py run_name=${1}
