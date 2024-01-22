
import os
import pickle
import sys

import ml_collections
import numpy as np
import torch
import pytorch_lightning as pl
from absl import flags, logging
from ml_collections import config_flags
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from models import infer_utils

logging.set_verbosity(logging.INFO)


def infer(config: ml_collections.ConfigDict, workdir: str = "./logging/"):

    pl.seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read in the processed dataset and prepare the data loader for training
    data_dir = os.path.join(config.data.root, config.data.name)
    data_processed_path = os.path.join(
        data_dir, f"processed/{config.name}.pkl")

    logging.info("Loading processed data from %s", data_processed_path)

    with open(data_processed_path, "rb") as f:
        data = pickle.load(f)
    _, data_loader, norm_dict = datasets.prepare_dataloader(
        data, train_frac=0.8, train_batch_size=1024, eval_batch_size=1024,
        num_workers=4, seed=config.seed)

    # load model
    model = regressor.Regressor.load_from_checkpoint(
        config.checkpoint_path, map_location=device)

    # start sampling
    samples, labels = infer_utils.sample(
        model, data_loader, config.infer.num_samples,
        return_labels=True, norm_dict=norm_dict)

    # save samples and labels
    samples_path = os.path.join(
        config.workdir, config.name)


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
