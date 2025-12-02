
import json
from ml_collections import config_dict

BEST_PARAMS_PATH = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/optuna/6d-uni-spline-sf1/optuna_results/best_params.json'

def read_best_params():
    with open(BEST_PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    return best_params

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed_data = 9731
    cfg.seed_training = 4563
    cfg.seed_infer = 19823

    # data configuration
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/preprocessed_datasets'
    cfg.data.name = '6d-uni-spline-sf1'
    cfg.data.num_datasets = 30
    cfg.data.start_dataset = 0
    cfg.data.input_dim = 6
    cfg.data.output_dim = 6

    # logging configuration
    cfg.workdir = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/trained-models/best-hyperparams'
    cfg.name = 'spline-sf1-6d-large-nodensity-v3'
    cfg.enable_progress_bar = True
    cfg.overwrite = True
    cfg.reset_optimizer = False
    cfg.checkpoint = None

    # read the best parameters
    best_params = read_best_params()

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.9
    cfg.train_batch_size = best_params['batch_size']
    cfg.num_workers = 0
    cfg.eval_batch_size = 128

    # prior configuration
    cfg.prior = config_dict.ConfigDict()
    cfg.prior.prior_min = [-4, -1, -100, -100, 0, 0, ]
    cfg.prior.prior_max = [-2, 0, 100, 100, 3, 360]

    # model configuration
    cfg.featurizer = config_dict.ConfigDict()
    cfg.featurizer.name = 'transformer'
    cfg.featurizer.d_feat_in = cfg.data.output_dim * 2
    cfg.featurizer.d_time_in = 1
    cfg.featurizer.d_feat = best_params['feat_embedding_dim']
    cfg.featurizer.d_time = best_params['feat_embedding_dim']
    cfg.featurizer.nhead = best_params['feat_nhead']
    cfg.featurizer.num_encoder_layers = best_params['feat_num_encoder']
    cfg.featurizer.dim_feedforward = best_params['feat_dim_feedforward']
    cfg.featurizer.batch_first = True
    cfg.featurizer.activation = config_dict.ConfigDict()
    cfg.featurizer.activation.name = 'Identity'
    cfg.mlp = config_dict.ConfigDict()
    cfg.mlp.hidden_sizes = [best_params['mlp_hidden_size']] * best_params['mlp_width']
    cfg.mlp.activation = config_dict.ConfigDict()
    cfg.mlp.activation.name = best_params['activation']
    cfg.mlp.batch_norm = best_params['mlp_batch_norm']
    # cfg.mlp.dropout = best_params['mlp_dropout']
    cfg.mlp.dropout = 0.5
    cfg.flows = config_dict.ConfigDict()
    cfg.flows.features = cfg.data.input_dim
    cfg.flows.hidden_sizes = [best_params['flows_hidden_size']] * best_params['flows_width']
    cfg.flows.num_transforms = best_params['flows_num_transforms']
    cfg.flows.num_bins = best_params['flows_num_bins']
    cfg.flows.activation = config_dict.ConfigDict()
    cfg.flows.activation.name = best_params['activation']

    # optimizer and scheduler configuration
    cfg.optimizer = config_dict.ConfigDict()
    cfg.optimizer.name = 'AdamW'
    cfg.optimizer.lr = best_params['lr']
    cfg.optimizer.betas = (0.9, 0.98)
    cfg.optimizer.weight_decay = best_params['weight_decay']
    cfg.optimizer.eps = 1e-9
    cfg.scheduler = config_dict.ConfigDict()
    cfg.scheduler.name = 'WarmUpCosineAnnealingLR'
    # cfg.scheduler.decay_steps = best_params['decay_steps']
    cfg.scheduler.decay_steps = 200_000
    cfg.scheduler.warmup_steps = best_params['warmup_steps']
    cfg.scheduler.eta_min = best_params['eta_min_factor']
    cfg.scheduler.interval = 'step'
    cfg.scheduler.restart = False

    # training loop configuration
    cfg.num_steps = cfg.scheduler.decay_steps
    cfg.patience = 100
    cfg.monitor = 'val_loss'
    cfg.mode = 'min'
    cfg.grad_clip = 0.5
    cfg.save_top_k = 5
    cfg.accelerator = 'gpu'

    return cfg
