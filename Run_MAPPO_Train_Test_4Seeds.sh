#!/bin/bash
#SBATCH --job-name=Train_Test
#SBATCH --account=project_2006417
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:a100:1,nvme:950

module --force purge
module load pytorch/1.12

pip3 install gym
pip3 install numpy-stl
pip3 install torch
pip3 install tqdm
pip3 install tensorboard 
pip3 install SciencePlots

python3 MAPPO_Run_Train_Test_8Seeds.py
