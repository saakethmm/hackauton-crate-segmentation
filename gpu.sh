#!/bin/bash
#SBATCH --job-name="GPUTest"
#SBATCH -p GPU
#SBATCH --gres=gpu:v100-32:8
#SBATCH --reservation=GPUcis230067p
#SBATCH -t 08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guntakan



#echo everything to stdout
set -x

#show gpu info
nvidia-smi
source hackauton/bin/activate
#load some software and run a script within your job
#https://www.psc.edu/resources/software/anaconda/

python3 finetune.py --bs 256 --net CRATE_base --opt adamW  --lr 5e-5 --n_epochs 200 --randomaug 1 --data cifar10 --data_dir data/
