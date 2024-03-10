
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from absl import flags, logging
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import io_utils, preprocess_utils

def calculate_derived_properties(table):
    ''' Calculate derived properties that are not stored in the dataset '''
    table['log_M_sat'] = np.log10(table['M_sat'])
    table['log_rs_sat'] = np.log10(table['rs_sat'])
    table['sin_phi'] = np.sin(table['phi'] / 360 * 2 * np.pi)
    table['cos_phi'] = np.cos(table['phi'] / 360 * 2 * np.pi)
    table['r_sin_phi'] = table['r'] * table['sin_phi']
    table['r_cos_phi'] = table['r'] * table['cos_phi']
    table['vz_abs'] = np.abs(table['vz'])
    table['vphi_abs'] = np.abs(table['vphi'])
    table['vtotal'] = np.sqrt(table['vphi']**2 + table['vz']**2)
    return table


def read_process_dataset(
    data_dir: Union[str, Path], labels: List[str], num_bins: int,
    phi1_min: float = None, phi1_max: float = None,
    num_datasets: int = 1, num_subsamples: int = 1,
    subsample_factor: int = 1, bounds: dict = None, 
    frac = False
):
    """ Read the dataset and preprocess

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the stream data.
    labels : list of str
        List of labels to use for the regression.
    num_bins : int
        Number of bins to use for binning the stream.
    phi1_min : float, optional
        Minimum value of phi1 to use. Default is None.
    phi1_max : float, optional
        Maximum value of phi1 to use. Default is None.
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    num_subsamples : int, optional
        Number of subsamples to use. Default is 1.
    subsample_factor : int, optional
        Factor to subsample the data. Default is 1.
    bounds : dict, optional
        Dictionary containing the bounds for each label. Default is None.
    frac: bool, optional
        If True, read datasets with two additional features:
        number and fraction of stars in each bin. 
        Default is False.
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
        table = calculate_derived_properties(table)

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

            # ignore out of bounds labels
            if bounds is not None:
                is_bound = True
                for key in bounds.keys():
                    lo, hi = bounds[key]
                    l = table[key].iloc[pid]
                    is_bound &= (l > lo) & (l < hi)
                if not is_bound:
                    continue
            label = table[labels].iloc[pid].values

            # TODO: figure out how to deal with t in the particle case
            for _ in range(num_subsamples):
                # subsample the stream
                phi1_subsample, feat_subsample = preprocess_utils.subsample_stream(
                    phi1, feat, subsample_factor=subsample_factor)

                if num_bins > 0:
                    # bin the stream
                    if frac:
                        phi1_bin_centers, feat_mean, feat_stdv, feat_count = preprocess_utils.bin_stream(
                        phi1_subsample, feat_subsample, num_bins=num_bins,
                        phi1_min=phi1_min, phi1_max=phi1_max, count=True)

                        feat_frac = feat_count/len(phi1)
                    else: 
                        phi1_bin_centers, feat_mean, feat_stdv = preprocess_utils.bin_stream(
                        phi1_subsample, feat_subsample, num_bins=num_bins,
                        phi1_min=phi1_min, phi1_max=phi1_max)

                    if len(phi1_bin_centers) == 0:
                        print('hello')
                        continue

                    if frac: 
                        x.append(np.concatenate([feat_mean, feat_stdv, feat_count, feat_frac], axis=1))
                    else: 
                        x.append(np.concatenate([feat_mean, feat_stdv], axis=1))
                    y.append(label)
                    t.append(phi1_bin_centers.reshape(-1, 1))
                else:
                    # TODO: figure out how to deal with t in the particle case
                    # no binning, particle-level data
                    x.append(feat_subsample)
                    y.append(label)
                    t.append(phi1_subsample.reshape(-1, 1))

    logging.info('Total number of samples: {}'.format(len(x)))

    x, padding_mask = preprocess_utils.pad_and_create_mask(x)
    t, _ = preprocess_utils.pad_and_create_mask(t)
    y = np.stack(y, axis=0)

    return x, y, t, padding_mask

def read_raw_dataset(
    data_dir: Union[str, Path], labels: List[str],
    phi1_min: float = None, phi1_max: float = None,
    num_datasets: int = 1
):
    """ Read raw data

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the stream data.
    labels : list of str
        List of labels to use for the regression.
    phi1_min : float, optional
        Minimum value of phi1 to use. Default is None.
    phi1_max : float, optional
        Maximum value of phi1 to use. Default is None.
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    """

    raw = []

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
        table = calculate_derived_properties(table)

        loop = tqdm(range(len(table)))

        for pid in loop:
            loop.set_description(f'Processing pid {pid}')
            phi1 = data['phi1'][pid]
            phi2 = data['phi2'][pid]
            pm1 = data['pm1'][pid]
            pm2 = data['pm2'][pid]
            vr = data['vr'][pid]
            dist = data['dist'][pid]

            raw.append(np.stack([phi1, phi2, pm1, pm2, vr, dist]))

    return raw

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

    # unpack the data and shuffle
    x, y, t, padding_mask = data
    num_train = int(train_frac * len(x))
    shuffle = np.random.permutation(len(x))
    x = x[shuffle]
    y = y[shuffle]
    t = t[shuffle]

    # normalize the data
    if norm_dict is None:
        # norm mask for x
        mask = np.repeat(~padding_mask[:num_train, :, None], x.shape[-1], axis=-1)
        x_loc = x[:num_train].mean(axis=(0, 1), where=mask)
        x_scale = x[:num_train].std(axis=(0, 1), where=mask)
        y_loc = y[:num_train].mean(axis=0)
        y_scale = y[:num_train].std(axis=0)
        # normalize time by min-max scaling
        t_loc = t[:num_train].min()
        t_scale = t[:num_train].max() - t_loc
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

    # convert to tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

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
