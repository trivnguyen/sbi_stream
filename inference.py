
import os
import sys
import glob
import h5py
sys.path.append('/global/u2/t/tvnguyen/snpe_stream')
import pickle
import yaml

import corner
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Uniform
import matplotlib.pyplot as plt
import matplotlib as mpl
import pytorch_lightning as pl
import numpy as np
from absl import flags, logging
from sbi.utils import BoxUniform
from ml_collections import ConfigDict, config_flags

from tqdm import tqdm

import datasets
from datasets import io_utils, preprocess_utils
from snpe_stream import infer_utils
from snpe_stream.npe import NPE

plt.style.use('/global/homes/t/tvnguyen/default.mplstyle')

def generate_seeds(base_seed, num_seeds, seed_range=(0, 2**32 - 1)):
    """Generate a list of RNG seeds deterministically from a base seed."""
    np.random.seed(base_seed)
    return np.random.randint(seed_range[0], seed_range[1], size=num_seeds, dtype=np.uint32)

def read_checkpoint(run_dir, run_name, round, device='cpu'):
    """ Read the checkpoint with the best val losses """
    checkpoint_dir = os.path.join(run_dir, run_name, f'round-{round}/lightning_logs/checkpoints/')
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f'Cannot find checkpoint dir {checkpoint_dir}')

    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, '*'))
    val_losses = []
    for path in sorted(all_checkpoints):
        val_losses.append(float(path.split('=')[-1].split('.ckpt')[0]))
    best_checkpoint =  all_checkpoints[np.argmin(val_losses)]

    print('Reading in checkpoint {}'.format(best_checkpoint))

    model = NPE.load_from_checkpoint(
        best_checkpoint, map_location=device).eval()
    return model, best_checkpoint


def main(config: ConfigDict):

    device = torch.device('cpu')
    if config.get('round') is None or config.get('round') < 0:
        raise ValueError('Please give round number')

    output_dir = os.path.join(
        config.workdir, config.name, f'round-{config.round}/proposal')
    os.makedirs(output_dir, exist_ok=True)

    # read in the data and preprocess
    data_processed_dir = os.path.join(
        root_processed, config.data.name, snpe_round_str, 'processed')
    data_processed_path = os.path.join(data_processed_dir, f"{name_processed}.pkl")
    os.makedirs(os.path.dirname(data_processed_path), exist_ok=True)

    data_raw_dir = os.path.join(config.data.root, config.data.name, snpe_round_str)

    if os.path.exists(data_processed_path):
        logging.info("Loading processed data from %s", data_processed_path)
        with open(data_processed_path, "rb") as f:
            data = pickle.load(f)
    else:
        logging.info("Processing raw data from %s", data_raw_dir)
        data = datasets.read_process_dataset(
            data_raw_dir,
            features=config.data.features,
            labels=config.data.labels,
            binning_fn=config.data.binning_fn,
            binning_args=config.data.binning_args,
            num_datasets=config.data.get("num_datasets", 1),
            num_subsamples=config.data.get("num_subsamples", 1),
            subsample_factor=config.data.get("subsample_factor", 1),
            bounds=config.data.get("label_bounds", None),
            frac=config.data.get('frac', False),
            use_width=config.data.get('use_width', True)
        )
        logging.info("Saving processed data to %s", data_processed_path)
        with open(data_processed_path, "wb") as f:
            pickle.dump(data, f)

    # Dataloader
    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        data,
        train_frac=config.train_frac,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        seed=data_seed  # reset seed for splitting train/val
    )

    # read in the best checkpoint
    model_embed, best_checkpoint = read_checkpoint(
        config.workdir, config.name, config.round, device=device)

    # Data compression
    with torch.no_grad():
        train_embed_x = []
        train_y = []
        for batch in train_loader:
            batch_dict = model_embed._prepare_training_batch(batch)
            x = batch_dict['x']
            t = batch_dict['t']
            y = batch_dict['y']
            padding_mask = batch_dict['padding_mask']
            embed_x = model(x, t, padding_mask=padding_mask)
            train_embed_x.append(embed_x.cpu().numpy())
            train_y.append(y.cpu().numpy())

        for batch in val_loader:
            batch_dict = model_embed._prepare_training_batch(batch)
            x = batch_dict['x']
            t = batch_dict['t']
            padding_mask = batch_dict['padding_mask']
            embed_x = model_embed(x, t, padding_mask=padding_mask)
            val_embed_x.append(embed_x.cpu().numpy())
            val_y.append(y.cpu().numpy())

    with h5py.File('') as f:



if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    main(config=FLAGS.config)

