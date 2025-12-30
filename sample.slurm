#!/bin/bash
#SBATCH --job-name=model_sample
#SBATCH --output=logs/sample_images_%j.out
#SBATCH --error=logs/sample_images_%j.err
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gres=gpu:1


echo "Job started on $(date)"
apptainer run --nv ../pml_updated.sif python run_sample_images.py
echo "Job finished on $(date)"
