#!/bin/bash
#SBATCH --job-name=tumor_class_5
#SBATCH -c 10
#SBATCH -p gpu
#SBATCH -G 1 
#SBATCH --time=04:00:00

ml reset
ml python/3.6.1
ml load viz
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load py-pytorch/1.6.0_py36
ml load py-scipy/1.1.0_py36
ml load py-scikit-learn/0.24.2_py36 
ml load py-numpy/1.19.2_py36

python3 pasnet/Run_Gene.py --config Config/Gene_Model.json

