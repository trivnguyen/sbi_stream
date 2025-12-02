import argparse
import os
import sys
import glob
sys.path.append('/global/u2/t/tvnguyen/sbi_stream')
import pickle

import tarp
import corner
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import astropy.constants as const
import astropy.units as u
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import datasets
from sbi.utils import BoxUniform
from sbi_stream import infer_utils
from sbi_stream.npe import NPE

parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True)
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--save_file', type=str, required=True)
args = parser.parse_args()

run_dir = args.run_dir
run_name = args.run_name
data_root = args.data_root
data_name = args.data_name
save_file = args.save_file

seed_data = 1031
num_datasets = 10
start_dataset = 36
batch_size = 512
seed = 319273

# Find the best checkpoint
checkpoint_list = sorted(glob.glob(os.path.join(run_dir, run_name, 'lightning_logs/checkpoints/*.ckpt')))
def get_val_loss(chkpt_path):
    fname = os.path.basename(chkpt_path)
    try:
        val_loss_str = fname.split('val_loss=')[-1].split('.ckpt')[0]
        return float(val_loss_str)
    except Exception:
        return float('inf')

best_checkpoint = min(checkpoint_list, key=get_val_loss)
checkpoint = os.path.basename(best_checkpoint)
print(f"Best checkpoint: {checkpoint}")

checkpoint_path = os.path.join(
    run_dir, run_name, 'lightning_logs/checkpoints/', checkpoint)
device = torch.device('cuda')
model = NPE.load_from_checkpoint(checkpoint_path, map_location=device)

# read dataset
data = datasets.read_processed_datasets(
    os.path.join(data_root, data_name),
    num_datasets=num_datasets, start_dataset=start_dataset,
)
_, loader, norm_dict = datasets.prepare_dataloader(
    data, train_frac=0.0001, train_batch_size=1, eval_batch_size=batch_size,
    num_workers=0, norm_dict=model.norm_dict, seed=seed_data,
    n_subsample=1,
    subsample_shuffle=False,
)

# inference
pl.seed_everything(seed)
samples, truths, log_probs = infer_utils.sample(
    model, loader, 1000, norm_dict=model.norm_dict, return_log_probs=True)
samples = samples.cpu().numpy()
truths = truths.cpu().numpy()
log_probs = log_probs.cpu().numpy()

# save the data
os.makedirs(os.path.dirname(save_file), exist_ok=True)
print(f"Saving results to {save_file}")
with open(save_file, 'wb') as f:
    pickle.dump({
        'samples': samples,
        'truths': truths,
        'log_probs': log_probs,
        'meta': {
            'data_root': data_root,
            'data_name': data_name,
            'num_datasets': num_datasets,
            'start_dataset': start_dataset,
            'seed_data': seed_data,
            'seed': seed,
            'checkpoint': checkpoint,
            'run_dir': run_dir,
            'run_name': run_name,
        }
    }, f)