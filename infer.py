
import os
import pickle
import sys

import ml_collections
import numpy as np
import torch
import pytorch_lightning as pl
from absl import flags, logging
from ml_collections import config_flags
from tqdm import tqdm

import datasets
from models.zuko import regressor as zuko_regressor
from models.zuko import infer_utils
from datasets import io_utils, preprocess_utils

logging.set_verbosity(logging.INFO)


def infer(config: ml_collections.ConfigDict, workdir: str = "./logging/"):

    pl.seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read in the processed dataset and prepare the data loader for training
    data_dir = os.path.join(config.data.root, config.data.name)
    logging.info("Processing raw data from %s", data_dir)

    data = datasets.read_process_dataset(
        data_dir,
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
    _, data_loader, norm_dict = datasets.prepare_dataloader(
        data,
        train_frac=config.train_frac,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )

    # load model
    checkpoint_path = os.path.join(
        config.workdir, config.name, 'lightning_logs/checkpoints', config.checkpoint
    )
    model = zuko_regressor.Regressor.load_from_checkpoint(
        checkpoint_path, map_location=device)

    samples, labels = infer_utils.sample(
        model, data_loader, config.infer.num_samples, return_labels=True, norm_dict=norm_dict)

    # save samples and labels
    samples_path = os.path.join(config.workdir, config.name, '6params-uni-zuko.pkl')
    with open(samples_path, 'wb') as f:
        data = {'samples': samples, 'labels': labels}
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    infer(config=FLAGS.config)
