#!/bin/bash
#SBATCH --job-name=pml_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus=1

echo "Job started on $(date)"
apptainer run --nv ../pml.sif python run_train_images.py
echo "Job finished on $(date)"
