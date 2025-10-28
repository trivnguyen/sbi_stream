
import os
import pickle
import sys
sys.path.append('/global/homes/t/tvnguyen/snpe_stream')
import shutil
import yaml
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import matplotlib as mpl
from absl import flags, logging
from ml_collections import ConfigDict, config_flags

import datasets


def preprocess(config: ConfigDict):

    workdir = os.path.join(config.root_out, config.name_out)

    # convert config to yaml and save
    os.makedirs(workdir, exist_ok=True)
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # read in the dataset and prepare the data loader for training
    data_raw_dir = os.path.join(config.root, config.name)

    logging.info("Processing raw data from %s", data_raw_dir)
    for i in tqdm(range(config.start_dataset, config.start_dataset + config.num_datasets)):
        data = datasets.read_raw_datasets(
            data_raw_dir,
            features=config.features,
            labels=config.labels,
            binning_fn=config.binning_fn,
            binning_args=config.binning_args,
            phi1_min=config.phi1_min,
            phi1_max=config.phi1_max,
            num_subsamples=config.get("num_subsamples", 1),
            subsample_factor=config.get("subsample_factor", 1),
            bounds=config.get("label_bounds", None),
            use_width=config.get('use_width', True),
            use_density=config.get('use_density', True),
            uncertainty=config.get('uncertainty', None),
            num_datasets=1,
            start_dataset=i,
        )
        if data is not None:
            data_out_path = os.path.join(workdir, f'data.{i}.pkl')
            logging.info("Saving processed data to %s", data_out_path)
            with open(data_out_path, "wb") as f:
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
    preprocess(config=FLAGS.config)
