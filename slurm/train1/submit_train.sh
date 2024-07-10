#!/bin/bash
#SBATCH -A m1727
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --job-name train1
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o out-train.out
#SBATCH -e err-train.err

# set up environment
module load conda
module load texlive/2022
conda activate sbi

config=$(realpath config.py)
cd /global/homes/r/rutong/sbi_stream

python train.py --config $config

exit 0
