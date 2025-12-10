#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH --output=logs/train_images_%j.out
#SBATCH --error=logs/train_images_%j.err
#SBATCH --partition=gpu-teaching-2d
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=head025
#SBATCH --gres=gpu:1


echo "Job started on $(date)"
apptainer run --nv ../pml_updated.sif python run_train_images.py
echo "Job finished on $(date)"
