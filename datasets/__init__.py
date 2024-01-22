
import os
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import utils

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

def write_dataset(path, data):
    """ Save dataset to HDF5 """
    # get the length of each sample and create a pointer
    sample_len = np.array([len(d) for d in data['phi1']])
    ptr = np.cumsum(sample_len)
    ptr = np.insert(ptr, 0, 0)

    # create the dataset
    with h5py.File(path, 'w') as f:
        for key in data.keys():
            f.create_dataset(
                key, data=np.concatenate(data[key]), compression='gzip')
        f.create_dataset('ptr', data=ptr, compression='gzip')

def read_dataset(path, unpack=True):
    """ Read dataset from HDF5 """
    data = {}
    with h5py.File(path, 'r') as f:
        ptr = f['ptr'][:]
        for key in f.keys():
            if key != 'ptr':
                val = f[key][:]
                if unpack:
                    val = np.split(val, ptr[1:-1])
                data[key] = val
    return data, ptr

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
    # read in the stream labels and data
    table = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    data, ptr = read_dataset(os.path.join(data_dir, 'data.hdf5'), unpack=True)

    # compute some derived labels
    table['log_M_sat'] = np.log10(table['M_sat'])

    loop = tqdm(range(len(table)))

    x, y, t = [], [], []
    for i in loop:
        loop.set_description(f'Processing pid {i}')
        phi1 = data['phi1'][i]
        phi2 = data['phi2'][i]
        pm1 = data['pm1'][i]
        pm2 = data['pm2'][i]
        vr = data['vr'][i]
        dist = data['dist'][i]
        feat = np.stack([phi2, pm1, pm2, vr, dist], axis=1)
        label = table[labels].iloc[i].values

        # bin the stream
        phi1_bin_centers, feat_mean, feat_stdv = utils.bin_stream(
            phi1, feat, num_bins=num_bins)

        x.append(np.concatenate([feat_mean, feat_stdv], axis=1))
        y.append(label)
        t.append(phi1_bin_centers.reshape(-1, 1))

    x, padding_mask = utils.pad_and_create_mask(x)
    t, _ = utils.pad_and_create_mask(t)
    y = np.stack(y, axis=0)

    return x, y, t, padding_mask

def prepare_dataloader(
    data: Tuple,
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


# def read_process_dataset(
#     data_dir: Union[str, Path], labels: List[str], num_bins: int):
#     """ Read and process the dataset.

#     Parameters
#     ----------
#     data_dir : str
#         Path to the directory containing the stream data.
#     labels : list of str
#         List of labels to use for the regression.
#     num_bins : int
#         Number of bins to use for binning the stream.
#     """
#     # read in the stream labels
#     table = pd.read_csv(os.path.join(data_dir, 'labels.csv'), delimiter=', ')
#     table = table[table['M_sat'] < 0.1]  # pick satellite below 10^9 Msun
#     table = table[~table['pid'].between(1700, 2199)]

#     # compute some derived labels
#     table['log_M_sat'] = np.log10(table['M_sat'])

#     loop = tqdm(range(len(table)))

#     x, y, t = [], [], []
#     for i in loop:
#         index = table['pid'].iloc[i]
#         loop.set_description(f'Processing pid {index}')

#         parameters, phi1, phi2, pm1, pm2, vr, dist = read_stream_index(
#             index, data_dir)
#         feat = np.stack([phi2, pm1, pm2, vr, dist], axis=1)
#         label = table[labels].iloc[i].values

#         # bin the stream
#         phi1_bin_centers, feat_mean, feat_stdv = bin_stream(
#             phi1, feat, num_bins=num_bins)

#         x.append(np.concatenate([feat_mean, feat_stdv], axis=1))
#         y.append(label)
#         t.append(phi1_bin_centers.reshape(-1, 1))

#     x, padding_mask = pad_and_create_mask(x)
#     t, _ = pad_and_create_mask(t)
#     y = np.stack(y, axis=0)

#     return x, y, t, padding_mask