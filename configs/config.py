
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed_data = 93179
    cfg.seed_training = 1923817
    cfg.seed_infer = 19823

    # data configuration
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/preprocessed_datasets'
    cfg.data.name = '6d-uni-spline-sf1'
    cfg.data.num_datasets = 36
    cfg.data.start_dataset = 0

    # logging configuration
    cfg.workdir = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/trained-models/'
    cfg.name = 'spline-sf1-6d-large-nodensity-2'
    cfg.enable_progress_bar = True
    cfg.overwrite = True
    cfg.reset_optimizer = False
    cfg.checkpoint = None

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.9
    cfg.train_batch_size = 128
    cfg.num_workers = 0
    cfg.eval_batch_size = 128

    # prior configuration
    cfg.prior = config_dict.ConfigDict()
    cfg.prior.prior_min = [-4, -1, -100, -100, 0, 0, ]
    cfg.prior.prior_max = [-2, 0, 100, 100, 3, 360]

    # model configuration
    cfg.featurizer = config_dict.ConfigDict()
    cfg.featurizer.name = 'transformer'
    cfg.featurizer.d_feat_in = 12
    cfg.featurizer.d_time_in = 1
    cfg.featurizer.d_feat = 16
    cfg.featurizer.d_time = 16
    cfg.featurizer.nhead = 4
    cfg.featurizer.num_encoder_layers = 4
    cfg.featurizer.dim_feedforward = 128
    cfg.featurizer.batch_first = True
    cfg.featurizer.activation = config_dict.ConfigDict()
    cfg.featurizer.activation.name = 'Identity'
    cfg.mlp = config_dict.ConfigDict()
    cfg.mlp.hidden_sizes = [128, 128]
    cfg.mlp.activation = config_dict.ConfigDict()
    cfg.mlp.activation.name = 'gelu'
    cfg.mlp.batch_norm = True
    cfg.mlp.dropout = 0.4
    cfg.flows = config_dict.ConfigDict()
    cfg.flows.features = 6
    cfg.flows.hidden_sizes = [128, 128, 128, 128]
    cfg.flows.num_transforms = 4
    cfg.flows.num_bins = 8
    cfg.flows.activation = config_dict.ConfigDict()
    cfg.flows.activation.name = 'gelu'

    # optimizer and scheduler configuration
    cfg.optimizer = config_dict.ConfigDict()
    cfg.optimizer.name = 'AdamW'
    cfg.optimizer.lr = 5e-4
    cfg.optimizer.betas = (0.9, 0.98)
    cfg.optimizer.weight_decay = 1e-4
    cfg.optimizer.eps = 1e-9
    cfg.scheduler = config_dict.ConfigDict()
    cfg.scheduler.name = 'WarmUpCosineAnnealingLR'
    cfg.scheduler.decay_steps = 5_000_000  # include warmup steps
    cfg.scheduler.warmup_steps = 250_000
    cfg.scheduler.eta_min = 0.01
    cfg.scheduler.interval = 'step'
    cfg.scheduler.restart = False

    # training loop configuration
    cfg.num_steps = 5_000_000
    cfg.patience = 100
    cfg.monitor = 'val_loss'
    cfg.mode = 'min'
    cfg.grad_clip = 0.5
    cfg.save_top_k = 5
    cfg.accelerator = 'gpu'

    return cfg
