#!/bin/bash -l
#SBATCH --job-name=GaussianModel
#SBATCH --partition=gpu-2h
#SBATCH --gres=gpu:40gb:1
#SBATCH --output=/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/%A-output.txt
#SBATCH --error=/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/%A-error.txt
#SBATCH --chdir=/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making

container=../pml.sif

apptainer run --nv $container \
python train_model.py --argflag1 3 --argflag2 -10 --argflag3 abc