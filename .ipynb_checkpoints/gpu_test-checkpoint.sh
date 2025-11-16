#!/bin/bash -l
#SBATCH --job-name=GPU_TEST
#SBATCH --partition=gpu-5h
#SBATCH --gpus=1
#SBATCH --output=gpu_test.out
#SBATCH --error=gpu_test.err

echo "Node:"
hostname

echo "GPU:"
nvidia-smi
