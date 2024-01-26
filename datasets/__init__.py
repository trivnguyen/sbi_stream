
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import io_utils, preprocess_utils


def read_process_binned_dataset(
    data_dir: Union[str, Path], labels: List[str], num_bins: int,
    num_datasets: int = 1
    ):
    """ Read the dataset and process into a binned dataset

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the stream data.
    labels : list of str
        List of labels to use for the regression.
    num_bins : int
        Number of bins to use for binning the stream.
    """
    x, y, t = [], [], []

    for i in range(num_datasets):
        label_fn = os.path.join(data_dir, f'labels.{i}.csv')
        data_fn = os.path.join(data_dir, f'data.{i}.hdf5')

        if os.path.exists(label_fn) & os.path.exists(data_fn):
            print('Reading in data from {}'.format(data_fn))
        else:
            print('Dataset {} not found. Skipping...'.format(i))
            continue

        # read in the data and label
        table = pd.read_csv(label_fn)
        data, ptr = io_utils.read_dataset(data_fn, unpack=True)

        # compute some derived labels
        table['log_M_sat'] = np.log10(table['M_sat'])
        table['log_rs_sat'] = np.log10(table['rs_sat'])

        loop = tqdm(range(len(table)))

        for pid in loop:
            loop.set_description(f'Processing pid {pid}')
            phi1 = data['phi1'][pid]
            phi2 = data['phi2'][pid]
            pm1 = data['pm1'][pid]
            pm2 = data['pm2'][pid]
            vr = data['vr'][pid]
            dist = data['dist'][pid]
            feat = np.stack([phi2, pm1, pm2, vr, dist], axis=1)
            label = table[labels].iloc[pid].values

            # bin the stream
            phi1_bin_centers, feat_mean, feat_stdv = preprocess_utils.bin_stream(
                phi1, feat, num_bins=num_bins)

            x.append(np.concatenate([feat_mean, feat_stdv], axis=1))
            y.append(label)
            t.append(phi1_bin_centers.reshape(-1, 1))

    x, padding_mask = preprocess_utils.pad_and_create_mask(x)
    t, _ = preprocess_utils.pad_and_create_mask(t)
    y = np.stack(y, axis=0)

    return x, y, t, padding_mask


def read_process_part_dataset(
    data_dir: Union[str, Path], labels: List[str]):
    """ Read dataset and process into a particle dataset """

    x, y, t = [], [], []

    for i in range(num_datasets):
        label_fn = os.path.join(data_dir, f'labels.{i}.csv')
        data_fn = os.path.join(data_dir, f'data.{i}.hdf5')

        if os.path.exists(label_fn) & os.path.exists(data_fn):
            print('Reading in data from {}'.format(data_fn))
        else:
            print('Dataset {} not found. Skipping...'.format(i))
            continue

        # read in the data and label
        table = pd.read_csv(label_fn)
        data, ptr = io_utils.read_dataset(data_fn, unpack=True)

        # compute some derived labels
        table['log_M_sat'] = np.log10(table['M_sat'])
        table['log_rs_sat'] = np.log10(table['rs_sat'])

        loop = tqdm(range(len(table)))

        for pid in loop:
            loop.set_description(f'Processing pid {pid}')
            phi1 = data['phi1'][pid]
            phi2 = data['phi2'][pid]
            pm1 = data['pm1'][pid]
            pm2 = data['pm2'][pid]
            vr = data['vr'][pid]
            dist = data['dist'][pid]
            feat = np.stack([phi2, pm1, pm2, vr, dist], axis=1)
            label = table[labels].iloc[pid].values

            x.append(feat)
            y.append(label)
            t.append(phi1_bin_centers.reshape(-1, 1))

    x, padding_mask = preprocess_utils.pad_and_create_mask(x)
    t, _ = preprocess_utils.pad_and_create_mask(t)
    y = np.stack(y, axis=0)

    return x, y, t, padding_mask



def prepare_dataloader(
    data: Tuple,
    norm_dict: dict = None,
    train_frac: float = 0.8,
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42
):
    """ Create dataloaders for training and evaluation. """
    pl.seed_everything(seed)

    # unpack the data and convert to tensor
    x, y, t, padding_mask = data
    num_train = int(train_frac * len(x))
    shuffle = np.random.permutation(len(x))
    x = torch.tensor(x[shuffle], dtype=torch.float32)
    y = torch.tensor(y[shuffle], dtype=torch.float32)
    t = torch.tensor(t[shuffle], dtype=torch.float32)
    padding_mask = torch.tensor(padding_mask[shuffle], dtype=torch.bool)

    # normalize the data
    if norm_dict is None:
        x_loc = x[:num_train].mean(dim=0)
        x_scale = x[:num_train].std(dim=0)
        y_loc = y[:num_train].mean(dim=0)
        y_scale = y[:num_train].std(dim=0)
        t_loc = t[:num_train].mean(dim=0)
        t_scale = t[:num_train].std(dim=0)
        norm_dict = {
            "x_loc": x_loc, "x_scale": x_scale,
            "y_loc": y_loc, "y_scale": y_scale,
            "t_loc": t_loc, "t_scale": t_scale
        }
    else:
        x_loc = norm_dict["x_loc"]
        x_scale = norm_dict["x_scale"]
        y_loc = norm_dict["y_loc"]
        y_scale = norm_dict["y_scale"]
        t_loc = norm_dict["t_loc"]
        t_scale = norm_dict["t_scale"]
    x = (x - x_loc) / x_scale
    y = (y - y_loc) / y_scale
    t = (t - t_loc) / t_scale

    # create data loader
    train_dset = TensorDataset(
        x[:num_train], y[:num_train], t[:num_train],
        padding_mask[:num_train])
    val_dset = TensorDataset(
        x[num_train:], y[num_train:], t[num_train:],
        padding_mask[num_train:])
    train_loader = DataLoader(
        train_dset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm_dict
