#!/bin/bash

#SBATCH --mem=224G
#SBATCH --gres=gpu:40g:1
#SBATCH --job-name mbrlyg
#SBATCH --cpus-per-task=3

python3 train_meta_model.py
