
import os
import pickle
import sys
sys.path.append('/global/homes/t/tvnguyen/sbi_stream')
import shutil
import yaml

import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import matplotlib as mpl
from absl import flags, logging
from sbi.utils import BoxUniform
from ml_collections import ConfigDict, config_flags

import datasets
from sbi_stream.npe import NPE


def generate_seeds(base_seed, num_seeds, seed_range=(0, 2**32 - 1)):
    """Generate a list of RNG seeds deterministically from a base seed."""
    np.random.seed(base_seed)
    return np.random.randint(seed_range[0], seed_range[1], size=num_seeds, dtype=np.uint32)


def train(config: ConfigDict):

    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, config.workdir))

    workdir = os.path.join(config.workdir, name)

    checkpoint_path = None
    if config.get('checkpoint') is not None:
        if os.path.isabs(config.checkpoint):
            checkpoint_path = config.checkpoint   # use full path
        else:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'Checkpoint {checkpoint_path} does not exist')
    logging.info(f'Reading checkpoint {checkpoint_path}')

    if os.path.exists(workdir):
        if checkpoint_path is None:
            if config.overwrite:
                shutil.rmtree(workdir)
            else:
                raise ValueError(
                    f"Workdir {workdir} already exists. Please set overwrite=True "\
                        "to overwrite the existing directory.")

    # convert config to yaml and save
    os.makedirs(workdir, exist_ok=True)
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # generate data and training seed
    data_seed = generate_seeds(config.seed_data, 10_000)[0]
    training_seed = generate_seeds(config.seed_training, 10_000)[0]

    # Determine norm_dict source
    if checkpoint_path is not None:
        model = NPE.load_from_checkpoint(checkpoint_path=checkpoint_path)
        norm_dict = model.norm_dict  # Use the norm_dict from the checkpointed model
        print(f'Loading from checkpoint {checkpoint_path}')
        print('Using norm_dict: ', norm_dict)
        if config.reset_optimizer:
            checkpoint_path = None
    else:
        model = None
        norm_dict = None

    # read in the dataset and prepare the data loader for training
    data = datasets.read_processed_datasets(
        os.path.join(config.data.root, config.data.name),
        num_datasets=config.data.num_datasets,
        start_dataset=config.data.get('start_dataset', 0),
    )
    # print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)

    # Prepare dataloader with the appropriate norm_dict
    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        data,
        train_frac=config.train_frac,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        norm_dict=norm_dict,
        seed=data_seed,
        n_subsample=config.n_subsample,
        subsample_shuffle=config.subsample_shuffle,
    )

    # If no checkpoint was loaded, initialize a new model with the determined norm_dict
    if model is None:
        model = NPE(
            featurizer_args=config.featurizer,
            flows_args=config.flows,
            mlp_args=config.mlp,
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            norm_dict=norm_dict,
            num_atoms=0,
            use_atomic_loss=False,
        )

    # Set prior
    prior_min = np.array(config.prior.prior_min)
    prior_max = np.array(config.prior.prior_max)
    prior_min_norm = (prior_min - norm_dict['y_loc']) / norm_dict['y_scale']
    prior_max_norm = (prior_max - norm_dict['y_loc']) / norm_dict['y_scale']
    prior_min_norm = torch.tensor(prior_min_norm, dtype=torch.float32)
    prior_max_norm = torch.tensor(prior_max_norm, dtype=torch.float32)
    prior = BoxUniform(
        low=prior_min_norm, high=prior_max_norm,
        device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    model.set_prior(prior)

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
        devices=1
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(training_seed)
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
    train(config=FLAGS.config)
