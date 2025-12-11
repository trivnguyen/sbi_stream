
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from absl import flags, logging
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import io_utils, preprocess_utils

DEFAULT_LABELS = ['log_M_sat', 'vz']
DEFAULT_FEATURES = ['phi2', 'pm1', 'pm2', 'vr', 'dist']


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

def read_raw_datasets(
    data_dir: Union[str, Path], features: List[str] = None, labels: List[str] = None,
    binning_fn: str = None, binning_args: dict = None, num_datasets: int = 1,
    start_dataset: int = 0, num_subsamples: int = 1, subsample_factor: int = 1,
    phi1_min: Optional[float] = None, phi1_max: Optional[float] = None,
    bounds: dict = None, use_density: bool = True, use_width: bool = True,
    uncertainty: Optional[str] = None,
):
    """ Read the dataset and preprocess

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the stream data.
    labels : list of str
        List of labels to use for the regression.
    binning_fn: str
        The binning function
    binning_args: dict
        Args of the binning function
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    start_dataset: int
        Index to start reading the dataset
    num_subsamples : int, optional
        Number of subsamples to use. Default is 1.
    subsample_factor : int, optional
        Factor to subsample the data. Default is 1.
    bounds : dict, optional
        Dictionary containing the bounds for each label. Default is None.
    use_density: bool, optional
        If True, use the fraction of stars in each bin
    use_width: bool, optional
        If True, use the width of the stream in each bin
    uncertainty: bool, optional,
        If not None, include uncertainty. Either "present" or "future"
    """
    # default args
    labels = labels or DEFAULT_LABELS
    features = features or DEFAULT_FEATURES
    binning_args = binning_args or {}

    x, y, t = [], [], []

    for i in range(start_dataset, start_dataset + num_datasets):
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
            feat = np.stack([data[f][pid] for f in features], axis=1)

            phi1_min = phi1_min or phi1.min()
            phi1_max = phi1_max or phi1.max()
            mask = (phi1_min <= phi1) & (phi1 < phi1_max)
            phi1 = phi1[mask]
            phi2 = phi2[mask]
            feat = feat[mask]

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
                phi1_subsample, phi2_subsample, feat_subsample = preprocess_utils.subsample_arrays(
                    [phi1, phi2, feat], subsample_factor=subsample_factor)
                phi1_subsample, phi2_subsample, feat_subsample, _ = preprocess_utils.add_uncertainty(
                    phi1_subsample, phi2_subsample, feat_subsample, features,
                    uncertainty=uncertainty)

                if binning_fn != 'particle':
                    # bin the stream
                    if binning_fn == 'bin_stream':
                        bin_centers, feat_mean, feat_stdv, feat_count = preprocess_utils.bin_stream(
                            phi1_subsample, feat_subsample, **binning_args)
                    elif binning_fn == 'bin_stream_spline':
                        bin_centers, feat_mean, feat_stdv, feat_count = preprocess_utils.bin_stream_spline(
                            phi1_subsample, phi2_subsample, feat_subsample, **binning_args)
                    elif binning_fn == 'bin_stream_hilmi24':
                        bin_centers, feat_mean, feat_stdv, feat_count = preprocess_utils.bin_stream_hilmi24(
                            phi1_subsample, phi2_subsample, feat_subsample, **binning_args)
                    if len(bin_centers) == 0:
                        continue

                    all_feats = []
                    all_feats.append(feat_mean)
                    if use_width:
                        all_feats.append(feat_stdv)
                    if use_density:
                        all_feats.append(feat_count / np.sum(feat_count))
                    all_feats = np.concatenate(all_feats, axis=1)
                    x.append(all_feats)
                    y.append(label)
                    t.append(bin_centers.reshape(-1, 1))
                else:
                    # no binning, particle-level data
                    # TODO: figure out how to deal with t in the particle case
                    mask = (binning_args.phi1_min < phi1_subsample) & (phi1_subsample < binning_args.phi1_max)
                    phi1_subsample = phi1_subsample[mask]
                    phi2_subsample = phi2_subsample[mask]
                    feat_subsample = feat_subsample[mask]
                    x.append(feat_subsample)
                    y.append(label)
                    t.append(phi1_subsample.reshape(-1, 1))

    logging.info('Total number of samples: {}'.format(len(x)))

    if len(x) > 0:
        x, padding_mask = preprocess_utils.pad_and_create_mask(x)
        t, _ = preprocess_utils.pad_and_create_mask(t)
        y = np.stack(y, axis=0)
        return x, y, t, padding_mask
    else:
        return None


def read_processed_datasets(
    data_dir: Union[str, Path], num_datasets: int = 1, start_dataset: int = 0,
):
    x, y, t, padding_mask = [], [], [], []
    xlen = []
    for i in tqdm(range(start_dataset, start_dataset + num_datasets)):
        data_path = os.path.join(data_dir, f'data.{i}.pkl')
        if not os.path.exists(data_path):
            continue
        with open(data_path, "rb") as f:
            d = pickle.load(f)
        x.append(d[0])
        y.append(d[1])
        t.append(d[2])
        padding_mask.append(d[3])
        xlen.append(d[0].shape[1])

    # Pad sequences to the maximum length
    # check if the lengths of sequences are the same
    if len(set(xlen)) != 1:
        max_xlen = max(xlen)
        for i in range(len(x)):
            pad_len = max_xlen - xlen[i]
            x[i] = np.pad(x[i], ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=0)
            t[i] = np.pad(t[i], ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=0)
            padding_mask[i] = np.pad(padding_mask[i], ((0, 0), (0, pad_len)), mode='constant', constant_values=True)

    x = np.concatenate(x)
    y = np.concatenate(y)
    t = np.concatenate(t)
    padding_mask = np.concatenate(padding_mask)

    return [x, y, t, padding_mask]

def prepare_dataloader(
    data: Tuple,
    norm_dict: dict = None,
    train_frac: float = 0.8,
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    n_subsample: int = 1,
    subsample_shuffle: bool = True,
):
    """ Create dataloaders for training and evaluation. """
    rng = np.random.default_rng(seed)

    # unpack the data
    x, y, t, padding_mask = data
    num_total = len(x)

    if subsample_shuffle & (n_subsample > 1):
        # special case if subsampling is enabled
        # this is required to prevent data leakage
        assert num_total % n_subsample == 0, f"Data size {len(x)} must be divisible by n_subsample {n_subsample}"
        num_total_subsample = num_total // n_subsample

        x = x.reshape(num_total_subsample, n_subsample, x.shape[1], x.shape[2])
        y = y.reshape(num_total_subsample, n_subsample, y.shape[1])
        t = t.reshape(num_total_subsample, n_subsample, t.shape[1], t.shape[2])
        padding_mask = padding_mask.reshape(num_total_subsample, n_subsample, padding_mask.shape[1])

        shuffle = rng.permutation(num_total_subsample)
        x = x[shuffle]
        y = y[shuffle]
        t = t[shuffle]
        padding_mask = padding_mask[shuffle]

        # split the data into training and validation sets
        num_train = int(train_frac * num_total_subsample)
        x_train, x_val = x[:num_train], x[num_train:]
        y_train, y_val = y[:num_train], y[num_train:]
        t_train, t_val = t[:num_train], t[num_train:]
        padding_mask_train, padding_mask_val = padding_mask[:num_train], padding_mask[num_train:]

        # flatten the data back to original shape
        x_train = x_train.reshape(-1, x_train.shape[2], x_train.shape[3])
        y_train = y_train.reshape(-1, y_train.shape[2])
        t_train = t_train.reshape(-1, t_train.shape[2], t_train.shape[3])
        padding_mask_train = padding_mask_train.reshape(-1, padding_mask_train.shape[2])
        x_val = x_val.reshape(-1, x_val.shape[2], x_val.shape[3])
        y_val = y_val.reshape(-1, y_val.shape[2])
        t_val = t_val.reshape(-1, t_val.shape[2], t_val.shape[3])
        padding_mask_val = padding_mask_val.reshape(-1, padding_mask_val.shape[2])
    else:
        shuffle = rng.permutation(num_total)
        x = x[shuffle]
        y = y[shuffle]
        t = t[shuffle]
        padding_mask = padding_mask[shuffle]

        num_train = int(train_frac * len(x))
        x_train, x_val = x[:num_train], x[num_train:]
        y_train, y_val = y[:num_train], y[num_train:]
        t_train, t_val = t[:num_train], t[num_train:]
        padding_mask_train, padding_mask_val = padding_mask[:num_train], padding_mask[num_train:]

    # normalize the data
    if norm_dict is None:
        # norm mask for x
        mask = np.repeat(~padding_mask_train[..., None], x_train.shape[-1], axis=-1)
        x_loc = x_train.mean(axis=(0, 1), where=mask)
        x_scale = x_train.std(axis=(0, 1), where=mask)

        # normalize y such that it is in range [-1, 1]
        # this is required for NSF
        y_min = np.min(y_train, axis=0)
        y_max = np.max(y_train, axis=0)
        y_loc = (y_min + y_max) / 2
        y_scale = (y_max - y_min) / 2

        # normalize time by min-max scaling
        t_loc = t_train.min()
        t_scale = t_train.max() - t_loc
        norm_dict = {
            "x_loc": x_loc, "x_scale": x_scale,
            "y_loc": y_loc, "y_scale": y_scale,
            "t_loc": t_loc, "t_scale": t_scale
        }
    else:
        x_loc, x_scale = norm_dict["x_loc"], norm_dict["x_scale"]
        y_loc, y_scale = norm_dict["y_loc"], norm_dict["y_scale"]
        t_loc, t_scale = norm_dict["t_loc"], norm_dict["t_scale"]

    # normalize the data
    x_train = (x_train - x_loc) / x_scale
    y_train = (y_train - y_loc) / y_scale
    t_train = (t_train - t_loc) / t_scale
    x_val = (x_val - x_loc) / x_scale
    y_val = (y_val - y_loc) / y_scale
    t_val = (t_val - t_loc) / t_scale

    # convert to tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    padding_mask_train = torch.tensor(padding_mask_train, dtype=torch.bool)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    t_val = torch.tensor(t_val, dtype=torch.float32)
    padding_mask_val = torch.tensor(padding_mask_val, dtype=torch.bool)

    # create data loader
    train_dset = TensorDataset(x_train, y_train, t_train, padding_mask_train)
    val_dset = TensorDataset(x_val, y_val, t_val, padding_mask_val)
    train_loader = DataLoader(
        train_dset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm_dict
