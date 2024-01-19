
import os
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def pad_and_create_mask(features, max_len=None):
    """ Pad and create Transformer mask. """
    if max_len is None:
        max_len = max([f.shape[0] for f in features])

    # create mask (batch_size, max_len)
    # NOTE: that jax mask is 1 for valid entries and 0 for padded entries
    # this is the opposite of the pytorch mask
    # here we are using the PyTorch mask
    mask = np.zeros((len(features), max_len), dtype=bool)
    for i, f in enumerate(features):
        mask[i, f.shape[0]:] = True

    # zero pad features
    padded_features = np.zeros((len(features), max_len, features[0].shape[1]))
    for i, f in enumerate(features):
        padded_features[i, :f.shape[0]] = f
    return padded_features, mask

def read_stream_index(index: int, root: Union[str, Path]):
    """ Read in the stream data for a given index. """
    path = os.path.join(root, f'sim{index}.hdf5')
    with h5py.File(path, 'r') as f:
        parameters = f['parameters'][:]
        phi1 =  f['phi1'][:]
        phi2 = f['phi2'][:]
        pm1 = f['pm1'][:]
        pm2 = f['pm2'][:]
        vr = f['vr'][:]
        dist = f['dist'][:]
    return parameters, phi1, phi2, pm1, pm2, vr, dist

def bin_stream(phi1: np.ndarray, feat: np.ndarray, num_bins: int):
    """ Bin the stream along the phi1 coordinates and compute the mean and stdv
    of the features in each bin. """

    phi1_min, phi2_max = phi1.min(), phi1.max()
    phi1_bins = np.linspace(phi1_min, phi2_max, num_bins + 1)
    phi1_bin_centers = 0.5 * (phi1_bins[1:] + phi1_bins[:-1])

    feat_mean = np.zeros((num_bins, feat.shape[1]))
    feat_stdv = np.zeros((num_bins, feat.shape[1]))
    for i in range(num_bins):
        mask = (phi1 >= phi1_bins[i]) & (phi1 <= phi1_bins[i + 1])
        if mask.sum() <= 1:
            continue
        feat_mean[i] = feat[mask].mean(axis=0)
        feat_stdv[i] = feat[mask].std(axis=0)

    # TODO: find a better to handle this case
    # remove bins with no data
    mask = (feat_stdv.sum(axis=1) != 0)
    phi1_bin_centers = phi1_bin_centers[mask]
    feat_mean = feat_mean[mask]
    feat_stdv = feat_stdv[mask]

    return phi1_bin_centers, feat_mean, feat_stdv

def read_process_dataset(
    data_dir: Union[str, Path], labels: List[str], num_bins: int):
    """ Read and process the dataset.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the stream data.
    labels : list of str
        List of labels to use for the regression.
    num_bins : int
        Number of bins to use for binning the stream.
    """
    # read in the stream labels
    table = pd.read_csv(os.path.join(data_dir, 'labels.csv'), delimiter=', ')
    table = table[table['M_sat'] < 0.1]  # pick satellite below 10^9 Msun
    table = table[~table['pid'].between(1700, 2199)]

    loop = tqdm(range(len(table)))

    x, y, t = [], [], []
    for i in loop:
        index = table['pid'].iloc[i]
        loop.set_description(f'Processing pid {index}')

        parameters, phi1, phi2, pm1, pm2, vr, dist = read_stream_index(
            index, data_dir)
        feat = np.stack([phi2, pm1, pm2, vr, dist], axis=1)
        label = table[labels].iloc[i].values

        # bin the stream
        phi1_bin_centers, feat_mean, feat_stdv = bin_stream(
            phi1, feat, num_bins=num_bins)

        x.append(np.concatenate([feat_mean, feat_stdv], axis=1))
        y.append(label)
        t.append(phi1_bin_centers.reshape(-1, 1))

    x, padding_mask = pad_and_create_mask(x)
    t, _ = pad_and_create_mask(t)
    y = np.stack(y, axis=0)

    return x, y, t, padding_mask


def prepare_dataloader(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    norm_dict: dict = None,
    train_frac: float = 0.8,
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
    num_workers: int = 0
):
    """ Create dataloaders for training and evaluation. """

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

