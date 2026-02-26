"""Matched-filter image-based dataset"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import io_utils, preprocess_utils
from rubin_stream import (
    config,
    isochrones,
    photometry,
    matched_filter,
    observation_model,
    flags,
    background,
    streams,
    pipelines,
)


# Channel names in order — used as keys for norm_dict and channel selection
_CHANNEL_KEYS = ('signal', 'bg_lsst', 'bg_roman')


def read_and_process_raw(
    data_dir: Union[str, Path],
    features: List[str],
    labels: List[str],
    num_datasets: int = 1,
    start_dataset: int = 0,
    num_subsamples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read and process stream dataset into matched-filter image-based dataset

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing the stream data.
    features : list of str
        List of feature names to extract.
    labels : list of str
        List of labels to use for the regression.
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    start_dataset : int, optional
        Index to start reading the dataset. Default is 0.
    num_subsamples : int, optional
        Number of subsamples to use. Default is 1.

    Returns
    -------
    signal_hists : np.ndarray, shape (N, H, W)
    bg_hists_lsst : np.ndarray, shape (N, H, W)
    bg_hists_roman : np.ndarray, shape (N, H, W)
    labels : np.ndarray, shape (N, n_labels)
    """
    # setting up default args
    survey_config_dict = config.DEFAULT_SURVEY_CONFIG
    matched_filter_dict = config.DEFAULT_MATCHED_FILTER_CONFIG
    default_data_dir = config.DEFAULT_DATA_DIR
    stream = "aau"
    mag_max = 25.25
    mag_min = 19.6
    surveys = ["lsst_r1p9"]

    stream_info = streams.get_stream_info(stream, mag_max, data_dir=default_data_dir)
    matched_filter_dict = matched_filter_dict.copy()
    matched_filter_dict["dist_mod"] = stream_info["distance_mod"]
    matched_filter_dict["age"] = stream_info["age"]
    matched_filter_dict["z"] = stream_info["z"]
    matched_filter_dict["nstar"] = stream_info["nstar"]
    matched_filter_dict["stream"] = stream
    matched_filter_dict["mag_min"] = mag_min
    matched_filter_dict["mag_max"] = mag_max

    # initalize isochrones and pipeline
    iso_dict = isochrones.load_isochrones(
        survey_config_dict,
        age=matched_filter_dict["age"],
        z=matched_filter_dict["z"],
        surveys=surveys,
    )
    pipeline = pipelines.MockMatchedFilterPipeline(
        matched_filter_dict,
        surveys,
        iso_dict,
        survey_config_dict=survey_config_dict,
        default_data_dir=default_data_dir,
    )

    all_signal_hists = []
    all_bg_hists_lsst = []
    all_bg_hists_roman = []
    all_labels = []

    for i in range(start_dataset, start_dataset + num_datasets):
        label_fn = os.path.join(data_dir, f'labels.{i}.csv')
        data_fn = os.path.join(data_dir, f'data.{i}.hdf5')

        if os.path.exists(label_fn) and os.path.exists(data_fn):
            print('Reading in data from {}'.format(data_fn))
        else:
            print('Dataset {} not found. Skipping...'.format(i))
            continue

        # read in the data and label
        table = pd.read_csv(label_fn)
        table = preprocess_utils.calculate_derived_properties(table)
        data, ptr = io_utils.read_dataset(data_fn, unpack=False)

        for j in tqdm(range(len(table)), desc='Processing streams'):
            sim_model = pd.DataFrame(
                {col: data[col][ptr[j]:ptr[j+1]] for col in features}
            )
            label = table[labels].iloc[j].values

            for _ in range(num_subsamples):
                sim_model_sub = pipelines.cut_and_sample_sim(sim_model, matched_filter_dict)
                signal_hist, bg_hist_lsst, bg_hist_roman = pipeline.run(
                    sim_model_sub, bg="lsst")

                all_signal_hists.append(signal_hist)
                all_bg_hists_lsst.append(bg_hist_lsst)
                all_bg_hists_roman.append(bg_hist_roman)
                all_labels.append(label)

    all_signal_hists = np.array(all_signal_hists)
    all_bg_hists_lsst = np.array(all_bg_hists_lsst)
    all_bg_hists_roman = np.array(all_bg_hists_roman)
    all_labels = np.array(all_labels)

    print('Total number of samples: {}'.format(len(all_signal_hists)))

    return all_signal_hists, all_bg_hists_lsst, all_bg_hists_roman, all_labels


def read_processed(
    data_dir: Union[str, Path],
    num_datasets: int = 1,
    start_dataset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read preprocessed matched-filter image-based datasets from pickle files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the dataset files.
    num_datasets : int, optional
        Number of datasets to read. Default is 1.
    start_dataset : int, optional
        Index of the first dataset to read. Default is 0.

    Returns
    -------
    signal_hists : np.ndarray, shape (N, H, W)
    bg_hists_lsst : np.ndarray, shape (N, H, W)
    bg_hists_roman : np.ndarray, shape (N, H, W)
    labels : np.ndarray, shape (N, n_labels)
    """
    all_signal_hists = []
    all_bg_hists_lsst = []
    all_bg_hists_roman = []
    all_labels = []

    for i in tqdm(range(start_dataset, start_dataset + num_datasets)):
        data_path = os.path.join(data_dir, f'dataset_{i}.pkl')
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        all_signal_hists.extend(data['signal_hists'])
        all_bg_hists_lsst.extend(data['bg_hists_lsst'])
        all_bg_hists_roman.extend(data['bg_hists_roman'])
        all_labels.extend(data['labels'])

    print('Total number of samples loaded: {}'.format(len(all_signal_hists)))

    return (
        np.array(all_signal_hists),
        np.array(all_bg_hists_lsst),
        np.array(all_bg_hists_roman),
        np.array(all_labels),
    )


def prepare_dataloaders(
    data: Tuple,
    norm_dict: dict = None,
    train_frac: float = 0.8,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    num_subsamples: int = 1,
    channels: Optional[List[str]] = None,
):
    """Create train/val dataloaders for matched-filter image datasets.

    Parameters
    ----------
    data : tuple
        ``(signal_hists, bg_hists_lsst, bg_hists_roman, labels)`` as returned
        by :func:`read_and_process_raw` or :func:`read_processed`.
    norm_dict : dict, optional
        Pre-computed normalization parameters. Computed from training data if
        ``None``. Expected keys: ``x_mean``, ``x_std``, ``y_loc``, ``y_scale``,
        ``channels``.
    train_frac : float, optional
        Fraction of data used for training. Default 0.8.
    train_batch_size : int, optional
        Training batch size. Default 32.
    eval_batch_size : int, optional
        Validation batch size. Default 32.
    num_workers : int, optional
        DataLoader worker processes. Default 0.
    seed : int, optional
        Random seed for shuffling. Default 42.
    num_subsamples : int, optional
        Number of subsamples per stream (prevents data leakage). Default 1.
    channels : list of str, optional
        Which histogram channels to stack as input. Any subset of
        ``['signal', 'bg_lsst', 'bg_roman']``. Defaults to all three.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, norm_dict)``
    """
    signal_hists, bg_hists_lsst, bg_hists_roman, labels = data

    if channels is None:
        channels = list(_CHANNEL_KEYS)

    # Stack into (N, C, H, W)
    channel_map = {
        'signal': signal_hists,
        'bg_lsst': bg_hists_lsst,
        'bg_roman': bg_hists_roman,
    }
    x = np.stack([channel_map[c] for c in channels], axis=1).astype(np.float32)
    y = labels.astype(np.float32)
    num_total = len(x)

    rng = np.random.default_rng(seed)

    # Shuffle and split, respecting subsamples to prevent data leakage
    if num_subsamples > 1:
        assert num_total % num_subsamples == 0, \
            f"Data size {num_total} must be divisible by num_subsamples {num_subsamples}"
        num_groups = num_total // num_subsamples

        x = x.reshape(num_groups, num_subsamples, *x.shape[1:])
        y = y.reshape(num_groups, num_subsamples, *y.shape[1:])

        shuffle = rng.permutation(num_groups)
        x, y = x[shuffle], y[shuffle]

        num_train = int(train_frac * num_groups)
        x_train = x[:num_train].reshape(-1, *x.shape[2:])
        y_train = y[:num_train].reshape(-1, *y.shape[2:])
        x_val = x[num_train:].reshape(-1, *x.shape[2:])
        y_val = y[num_train:].reshape(-1, *y.shape[2:])
    else:
        shuffle = rng.permutation(num_total)
        x, y = x[shuffle], y[shuffle]

        num_train = int(train_frac * num_total)
        x_train, x_val = x[:num_train], x[num_train:]
        y_train, y_val = y[:num_train], y[num_train:]

    # Compute normalization from training data if not provided
    if norm_dict is None:
        # Per-channel stats across all training pixels: mean over (N, H, W) per channel
        x_mean = x_train.mean(axis=(0, 2, 3), keepdims=True).squeeze(0)  # (C, 1, 1)
        x_std = x_train.std(axis=(0, 2, 3), keepdims=True).squeeze(0)    # (C, 1, 1)
        x_std = np.where(x_std == 0, 1.0, x_std)  # guard against zero std

        y_min = y_train.min(axis=0)
        y_max = y_train.max(axis=0)
        y_loc = (y_min + y_max) / 2
        y_scale = (y_max - y_min) / 2

        norm_dict = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_loc': y_loc,
            'y_scale': y_scale,
            'channels': channels,
        }
    else:
        x_mean = norm_dict['x_mean']
        x_std = norm_dict['x_std']
        y_loc = norm_dict['y_loc']
        y_scale = norm_dict['y_scale']

    # Normalize — broadcasting (C, 1, 1) against (N, C, H, W)
    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_loc) / y_scale
    x_val = (x_val - x_mean) / x_std
    y_val = (y_val - y_loc) / y_scale

    # Convert to tensors and create DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, norm_dict


def prepare_test_dataloader(
    data: Tuple,
    norm_dict: dict,
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create a test dataloader for matched-filter image datasets.

    Parameters
    ----------
    data : tuple
        ``(signal_hists, bg_hists_lsst, bg_hists_roman, labels)``
    norm_dict : dict
        Normalization parameters from the training run.
    batch_size : int, optional
        Default 32.
    num_workers : int, optional
        Default 0.

    Returns
    -------
    DataLoader
    """
    signal_hists, bg_hists_lsst, bg_hists_roman, labels = data
    channels = norm_dict.get('channels', list(_CHANNEL_KEYS))

    x = _stack_channels(signal_hists, bg_hists_lsst, bg_hists_roman, channels)
    y = labels.astype(np.float32)

    x = (x - norm_dict['x_mean']) / norm_dict['x_std']
    y = (y - norm_dict['y_loc']) / norm_dict['y_scale']

    return DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
