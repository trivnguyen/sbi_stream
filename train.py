
import os
import pickle
import sys

import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags

import datasets
from models import models, regressor

logging.set_verbosity(logging.INFO)

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set seed
    pl.seed_everything(config.seed)

    # read in the dataset and prepare the data loader for training
    data_dir = os.path.join(config.data.root, config.data.name)
    data_processed_path = os.path.join(data_dir, f"processed/{config.name}.pkl")
    os.makedirs(os.path.dirname(data_processed_path), exist_ok=True)

    if os.path.exists(data_processed_path):
        logging.info("Loading processed data from %s", data_processed_path)
        with open(data_processed_path, "rb") as f:
            data = pickle.load(f)
    else:
        logging.info("Processing raw data from %s", data_dir)
        data = datasets.read_process_dataset(
            data_dir, config.data.labels, config.data.num_bins)
        logging.info("Saving processed data to %s", data_processed_path)
        with open(data_processed_path, "wb") as f:
            pickle.dump(data, f)

    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        data,
        train_frac=config.train_frac,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
    )

    # create model
    model = regressor.Regressor(
        output_size=config.output_size,
        featurizer_args=config.featurizer,
        flows_args=config.flows,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
    )

    # create the trainer object
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.monitor, patience=config.patience, mode=config.mode,
            verbose=True),
        pl.callbacks.ModelCheckpoint(
            monitor=config.monitor, save_top_k=config.save_top_k,
            mode=config.mode, save_weights_only=False),
        pl.callbacks.LearningRateMonitor("epoch"),
    ]

    train_logger = pl_loggers.TensorBoardLogger(workdir, name=config.name)
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.num_epochs,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    trainer.fit(
        model, train_loader, val_loader,
        ckpt_path=config.get("checkpoint_path", None),
    )

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
