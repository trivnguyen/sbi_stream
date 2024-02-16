#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --job-name train_0
#SBATCH --time 5:00:00
#SBATCH --gpus 1 -C v100
#SBATCH --mail-type all
#SBATCH --mail-user tnguy@mit.edu
#SBATCH -o out-train.out
#SBATCH -e err-train.err

# set up environment
module unload python
if [ -f "/mnt/home/tnguyen/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/mnt/home/tnguyen/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="/mnt/home/tnguyen/miniconda3/bin:$PATH"
fi
conda activate geometric

config=$(realpath config.py)
cd /mnt/home/tnguyen/projects/sbi_stream

python train.py --config $config

exit 0
