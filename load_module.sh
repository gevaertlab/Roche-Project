#!/bin/bash
#SBATCH --job-name=GO
#SBATCH -c 10
#SBATCH -p gpu
#SBATCH -G 1 
#SBATCH --time=30:00:00

ml reset
ml python/3.6.1
ml load py-pytorch/1.6.0_py36
ml load py-scipy/1.1.0_py36
ml load py-scikit-learn/0.19.1_py36
ml load py-matplotlib/3.2.1_py36

#python3 pasnet/Run.py
python3 pasnet/interpret.py
