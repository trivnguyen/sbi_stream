import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from absl import flags, logging
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import io_utils, preprocess_utils

def bin_stream(
    phi1: np.ndarray, feat: np.ndarray, num_bins: int,
    phi1_min: float = None, phi1_max: float = None
):
    """ Bin the stream along the phi1 coordinates and compute the mean and stdv
    of the features in each bin. """

    phi1_min = phi1_min or phi1.min()
    phi1_max = phi1_max or phi1.max()
    phi1_bins = np.linspace(phi1_min, phi1_max, num_bins + 1)
    phi1_bin_centers = 0.5 * (phi1_bins[1:] + phi1_bins[:-1])

    feat_mean = np.zeros((num_bins, feat.shape[1]))
    feat_stdv = np.zeros((num_bins, feat.shape[1]))
    feat_num = np.zeros((num_bins, 1))

    for i in range(num_bins):
        mask = (phi1 >= phi1_bins[i]) & (phi1 <= phi1_bins[i + 1])
        if mask.sum() <= 1:
            continue
        feat_mean[i] = feat[mask].mean(axis=0)
        feat_stdv[i] = feat[mask].std(axis=0)
        feat_num[i] = mask.sum()

    return phi1_bin_centers, feat_mean, feat_stdv, feat_num

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