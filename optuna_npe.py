
import os
import pickle
import sys
import json
import functools
sys.path.append('/global/homes/t/tvnguyen/sbi_stream')
import shutil
import yaml

import optuna
import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import matplotlib as mpl
from absl import flags, logging
from sbi.utils import BoxUniform
from ml_collections import ConfigDict, config_flags
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import FixedTrial

import tarp
import datasets
from sbi_stream.npe import NPE
from sbi_stream import infer_utils


def objective(trial, config):

    # Model hyperparameters
    activation_name = trial.suggest_categorical("activation", ["gelu", "silu", "relu"])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    feat_embedding_dim = trial.suggest_categorical("feat_embedding_dim", [8, 16, 32, 64])
    feat_nhead = trial.suggest_categorical("feat_nhead", [1, 2, 4, 8])
    feat_num_encoder = trial.suggest_int("feat_num_encoder", 2, 6)
    feat_dim_feedforward = trial.suggest_categorical("feat_dim_feedforward", [32, 64, 128, 256])
    mlp_hidden_size = trial.suggest_categorical("mlp_hidden_size", [32, 64, 128, 256])
    mlp_width = trial.suggest_int("mlp_width", 2, 6)
    # mlp_dropout = trial.suggest_float("mlp_dropout", 0.0, 0.8, step=0.1)
    mlp_dropout = trial.suggest_float("mlp_dropout", 0.0, 0.4, step=0.1)
    mlp_batch_norm = trial.suggest_categorical("mlp_batch_norm", [True, False])
    flows_hidden_size = trial.suggest_categorical("flows_hidden_size", [32, 64, 128, 256])
    flows_width = trial.suggest_int("flows_width", 2, 6)
    # flows_num_bins = trial.suggest_int('flows_num_bins', 4, 16)
    flows_num_bins = trial.suggest_int('flows_num_bins', 10, 16)
    flows_num_transforms = trial.suggest_int("flows_num_transforms", 2, 8)

    # Define hyperparameters to optimize
    featurizer_args = ConfigDict(dict(
        name='transformer',
        d_feat_in=config.data.d_feat_in,
        d_time_in=1,
        d_feat=feat_embedding_dim,
        d_time=feat_embedding_dim,
        nhead=feat_nhead,
        num_encoder_layers=feat_num_encoder,
        dim_feedforward=feat_dim_feedforward,
        batch_first=True,
        activation=ConfigDict(dict(name='Identity')),
    ))
    mlp_args = ConfigDict(dict(
        hidden_sizes=[mlp_hidden_size] * mlp_width,
        batch_norm=mlp_batch_norm,
        dropout=mlp_dropout,
        activation=ConfigDict(dict(name=activation_name)),
    ))
    flows_args = ConfigDict(dict(
        features=6,
        num_bins=flows_num_bins,
        hidden_sizes=[flows_hidden_size] * flows_width,
        num_transforms=flows_num_transforms,
        activation=ConfigDict(dict(name=activation_name)),
    ))

    # Optimizer hyperparameters
    # lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    lr = trial.suggest_float('lr', 2e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    optimizer_args = ConfigDict(dict(
        name='AdamW',
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-9,
        betas=(0.9, 0.98),
    ))

    # Scheduler hyperparameters
    warmup_steps = trial.suggest_int('warmup_steps', 1000, 10_000)
    decay_steps = trial.suggest_int('decay_steps', 100_000, 250_000)
    eta_min_factor = trial.suggest_float('eta_min_factor', 0.001, 0.1, log=True)
    scheduler_args = ConfigDict(dict(
        name='WarmUpCosineAnnealingLR',
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        eta_min=eta_min_factor,
        interval='step'
    ))

    # read in the dataset and prepare the data loader for training
    data = datasets.read_processed_datasets(
        os.path.join(config.data.root, config.data.name),
        num_datasets=config.data.num_datasets,
        start_dataset=config.data.get('start_dataset', 0),
    )
    # test purpose only
    # data[0] = data[0][:100]
    # data[1] = data[1][:100]
    # data[2] = data[2][:100]
    # data[3] = data[3][:100]

    # Prepare dataloader with the appropriate norm_dict
    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        data,
        train_frac=config.train_frac,
        train_batch_size=batch_size,
        eval_batch_size=128,
        num_workers=0,
        norm_dict=None,
        seed=config.seed_data,
    )
    model = NPE(
        featurizer_args=featurizer_args,
        flows_args=flows_args,
        mlp_args=mlp_args,
        optimizer_args=optimizer_args,
        scheduler_args=scheduler_args,
        norm_dict=norm_dict,
        num_atoms=0,
        use_atomic_loss=False,
    )

    # Create callbacks with Optuna pruning but without checkpointing during trials
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=100,
            mode='min',
            verbose=True
        ),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    # pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # Minimal logger to reduce overhead
    train_logger = pl_loggers.TensorBoardLogger(
        os.path.join(config.workdir, 'optuna_logs'), name=f"trial_{trial.number}")

    # Create trainer with properly separated callbacks
    trainer = pl.Trainer(
        default_root_dir=os.path.join(config.workdir, 'optuna_logs'),
        max_epochs=100,
        max_steps=decay_steps,
        accelerator='gpu',
        devices=1,
        callbacks=callbacks,
        logger=train_logger,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    # trainer.callbacks.append(pruning_callback)

    # Log all hyperparameters to the logger
    hyperparameters = {
        # Model hyperparameters
        'activation': activation_name,

        # Featurizer parameters
        'feat_embedding_dim': feat_embedding_dim,
        'feat_nhead': feat_nhead,
        'feat_num_encoder': feat_num_encoder,
        'feat_dim_feedforward': feat_dim_feedforward,

        # MLP parameters
        'mlp_hidden_size': mlp_hidden_size,
        'mlp_width': mlp_width,
        'mlp_dropout': mlp_dropout,
        'mlp_batch_norm': mlp_batch_norm,

        # Flows parameters
        'flows_hidden_size': flows_hidden_size,
        'flows_width': flows_width,
        'flows_num_transforms': flows_num_transforms,
        'flows_num_bins': flows_num_bins,

        # Optimizer parameters
        'lr': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,

        # Scheduler parameters
        'warmup_steps': warmup_steps,
        'decay_steps': decay_steps,
        'eta_min_factor': eta_min_factor,

        # Trial info
        'trial_number': trial.number,
    }
    trainer.logger.log_hyperparams(hyperparameters)

    # train the model
    logging.info("Training model...")
    pl.seed_everything()
    trainer.fit(model, train_loader, val_loader)


    # after training, evaluate the tarp curve
    samples, labels = infer_utils.sample(
        model, val_loader, num_samples=500, return_labels=True, return_log_probs=False,
        norm_dict=model.norm_dict)
    samples = samples.cpu().numpy()
    labels = labels.cpu().numpy()

    ecp, alpha = tarp.get_tarp_coverage(
        np.transpose(samples, [1, 0, 2]), labels, references="random", metric="euclidean",
        norm=True, bootstrap=True, num_bootstrap=100)
    tarp_val = torch.mean(torch.from_numpy(ecp[:, ecp.shape[1] // 2])).to(model.device)

    return trainer.callback_metrics['val_loss'].item(), abs(tarp_val - 0.5)


def main():
    # Parse command line arguments
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    config = FLAGS.config

    if config.overwrite:
        if os.path.exists(config.workdir):
            shutil.rmtree(config.workdir)
    os.makedirs(config.workdir, exist_ok=True)

    # # Pruner
    # pruner = optuna.pruners.MedianPruner(
    #     n_startup_trials=config.optuna.n_startup_trials,
    #     n_warmup_steps=config.optuna.n_warmup_steps,
    #     interval_steps=config.optuna.interval_steps
    # )

    # Create storage path with workdir
    storage_path = config.optuna.get("storage", "sqlite:///optuna_studies.db")

    # If it's a relative path (not starting with sqlite://, mysql://, postgresql://)
    if not storage_path.startswith(("sqlite://", "mysql://", "postgresql://")):
        # Check if the path is absolute or relative
        if not os.path.isabs(storage_path):
            # If it's a relative path, prepend the workdir
            storage_file = os.path.join(config.workdir, storage_path)
            # Make sure the directory exists
            os.makedirs(os.path.dirname(storage_file), exist_ok=True)
            # Create the SQLite URI
            storage_path = f"sqlite:///{storage_file}"
        else:
            # It's an absolute path, just convert to SQLite URI
            storage_path = f"sqlite:///{storage_path}"

    # Create the study
    study = optuna.create_study(
        directions=['minimize', 'minimize'],
        # pruner=pruner,
        study_name=config.optuna.get("study_name", "npe_hyperparameter_optimization"),
        storage=storage_path,
        load_if_exists=config.optuna.get("load_if_exists", True)
    )

    # Run the optimization
    objective_with_config = functools.partial(objective, config=config)
    study.optimize(
        objective_with_config,
        n_trials=config.optuna.get("n_trials", 100),
        timeout=config.optuna.get("timeout", None)
    )

    print("\nBest trial:")
    best = study.best_trials[0]
    print(f"  Log-prob: {best.values[0]:.4f}, TARP dev: {best.values[1]:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save best parameters to file for later use
    output_dir = os.path.join(config.workdir, 'optuna_results')
    os.makedirs(output_dir, exist_ok=True)

    best_params_path = os.path.join(output_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"Best parameters saved to '{best_params_path}'")


if __name__ == "__main__":
    main()
