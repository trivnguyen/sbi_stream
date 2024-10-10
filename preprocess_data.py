
import os
import pickle
import sys
import shutil

import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags

import datasets

logging.set_verbosity(logging.INFO)

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # read in the dataset and prepare the data loader for training
    data_processed_dir = os.path.join(
        config.data.root_processed, config.data.name, 'processed')
    data_processed_path = os.path.join(
        data_processed_dir, f"{config.data.name_processed}.pkl")
    os.makedirs(os.path.dirname(data_processed_path), exist_ok=True)

    data_raw_dir = os.path.join(config.data.root, config.data.name)

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
            frac=config.data.get('frac', False)
        )
        logging.info("Saving processed data to %s", data_processed_path)
        with open(data_processed_path, "wb") as f:
            pickle.dump(data, f)

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
    train(config=FLAGS.config, workdir=FLAGS.config.workdir)
