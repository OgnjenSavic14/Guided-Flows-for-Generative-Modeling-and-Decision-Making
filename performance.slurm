#!/bin/bash
#SBATCH --job-name=model_performance
#SBATCH --output=logs/performance_images_%j.out
#SBATCH --error=logs/performance_images_%j.err
#SBATCH --partition=gpu-teaching-2d
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=head025

echo "Job started on $(date)"
apptainer run --nv ../pml_updated.sif python run_performance_evaluation.py
echo "Job finished on $(date)"
