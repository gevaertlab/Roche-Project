#!/bin/bash
#SBATCH --job-name=GO
#SBATCH -c 10
#SBATCH -p gpu
#SBATCH -G 1 
#SBATCH --time=30:00:00

ml reset
ml python/3.6.1
ml load py-pandas/1.0.3_py36
ml load py-scipy/1.4.1_py36
ml load py-scikit-learn/0.24.2_py36
ml load viz
ml load py-matplotlib/3.2.1_py36