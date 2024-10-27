#!/bin/bash
#SBATCH -A m1727   # account number
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --job-name training
#SBATCH --time 8:00:00
#SBATCH --mail-type all
#SBATCH --mail-user # REMOVE THIS OR ADD YOUR EMAIL
#SBATCH -o log.out

# set up environment
source /pscratch/sd/t/tvnguyen/envs/sbi-stream/bin/activate

### CHANGE ALL OF THIS
# use absolute path to make sure that there is no bug
basedir = /path/to/sbi_stream  # this should be where repo is
config = /path/to/config.py

# cd to base directory and run the training script
cd $basedir
python train.py --config $config
exit 0