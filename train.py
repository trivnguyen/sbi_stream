
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
from models.zuko import regressor as zuko_regressor
from models import models, regressor, utils

logging.set_verbosity(logging.INFO)

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, workdir))

    workdir = os.path.join(workdir, name)
    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")

    # convert config to yaml and save
    os.makedirs(workdir, exist_ok=True)
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

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
            frac=config.data.get('frac', False),
            use_width=config.data.get('use_width', True)
        )
        logging.info("Saving processed data to %s", data_processed_path)
        with open(data_processed_path, "wb") as f:
            pickle.dump(data, f)

    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        data,
        train_frac=config.train_frac,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        seed=config.seed_data,  # reset seed for splitting train/val
    )

    # create model
    if config.flows.zuko:
        model = zuko_regressor.Regressor(
            output_size=config.output_size,
            featurizer_args=config.featurizer,
            flows_args=config.flows,
            mlp_args=config.mlp,
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            norm_dict=norm_dict,
        )
    else:
        model = regressor.Regressor(
            output_size=config.output_size,
            featurizer_args=config.featurizer,
            flows_args=config.flows,
            mlp_args=config.mlp,
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
            filename='{epoch}-{step}-{train_loss:.4f}-{val_loss:.4f}',
            mode=config.mode, save_weights_only=False),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.seed_training)
    trainer.fit(
        model, train_loader, val_loader,
        ckpt_path=checkpoint_path
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
