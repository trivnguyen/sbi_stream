
import os
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed_data = 93179

    # data configuration
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/preprocessed_datasets'
    cfg.data.name = '6d-uni-spline-sf1'
    cfg.data.num_datasets = 6
    cfg.data.start_dataset = 30
    cfg.data.d_feat_in = 12

    # logging configuration
    cfg.workdir = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/optuna/6d-uni-spline-sf1'
    cfg.overwrite = True

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.8
    cfg.num_workers = 0

    # optuna configuration
    cfg.optuna = config_dict.ConfigDict()
    cfg.optuna.n_startup_trials = 5
    cfg.optuna.n_warmup_steps = 10
    cfg.optuna.interval_steps = 10
    cfg.optuna.n_trials = 100
    cfg.optuna.study_name = "npe_hyperparameter_optimization"
    cfg.optuna.load_if_exists = True
    cfg.optuna.storage = "sqlite:///" + os.path.join(cfg.workdir, 'optuna_studies.db')

    return cfg
