
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed_data = 10
    cfg.seed_training = 20
    cfg.seed_infer = 30
    cfg.round = -1

    # data configuration
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/snpe/datasets/apt-runs'
    cfg.data.name = 'spline-sf1-6d'
    cfg.data.num_datasets = 20
    cfg.data.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r', 'phi']
    cfg.data.features = ['phi1', 'phi2', 'pm1', 'pm2', 'vr', 'dist']
    cfg.data.subsample_factor = 1
    cfg.data.num_subsamples = 1
    cfg.data.frac = True
    cfg.data.use_width = True
    cfg.data.binning_fn = 'bin_stream_spline'
    cfg.data.binning_args = binning = config_dict.ConfigDict()
    binning.num_bins = 50
    binning.num_knots = 50
    binning.phi1_min = -20
    binning.phi1_max = 12

    # data for inference
    cfg.data_target = config_dict.ConfigDict()
    cfg.data_target.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/datasets/'
    cfg.data_target.name = 'aau-ta25'
    cfg.data_target.num_samples = 20_000
    cfg.data_target.num_samples_batch = 10_000
    cfg.data_target.epsilon = 1e-3

    # logging configuration
    cfg.workdir = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/snpe/trained-models/apt-runs/'
    cfg.name = 'spline-sf1-6d'
    cfg.enable_progress_bar = True
    cfg.overwrite = True
    cfg.reset_optimizer = True
    cfg.checkpoint = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/snpe/trained-models/apt-runs/spline-sf1-6d/round-0/lightning_logs/checkpoints/epoch=357-step=467190-train_loss=-0.7129-val_loss=-1.4743.ckpt'

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.9
    cfg.train_batch_size = 64
    cfg.num_workers = 0
    cfg.eval_batch_size = 64

    # prior configuration
    cfg.prior = config_dict.ConfigDict()
    cfg.prior.prior_min = [-4, -1, -100, -100, 0, 0, ]
    cfg.prior.prior_max = [-2, 0, 100, 100, 3, 360]

    # model configuration
    cfg.use_atomic_loss = False
    cfg.num_atoms = 0
    if cfg.data.use_width:
        d_feat_in = len(cfg.data.features) * 2 + int(cfg.data.frac)
    else:
        d_feat_in  = len(cfg.data.features) + int(cfg.data.frac)
    cfg.featurizer = config_dict.ConfigDict()
    cfg.featurizer.name = 'transformer'
    cfg.featurizer.d_feat_in = d_feat_in
    cfg.featurizer.d_time_in = 1
    cfg.featurizer.d_feat = 16
    cfg.featurizer.d_time = 16
    cfg.featurizer.nhead = 4
    cfg.featurizer.num_encoder_layers = 4
    cfg.featurizer.dim_feedforward = 64
    cfg.featurizer.batch_first = True
    cfg.featurizer.activation = config_dict.ConfigDict()
    cfg.featurizer.activation.name = 'Identity'
    cfg.mlp = config_dict.ConfigDict()
    cfg.mlp.hidden_sizes = [64, 64]
    cfg.mlp.activation = config_dict.ConfigDict()
    cfg.mlp.activation.name = 'gelu'
    cfg.mlp.batch_norm = True
    cfg.mlp.dropout = 0.4
    cfg.flows = config_dict.ConfigDict()
    cfg.flows.features = len(cfg.data.labels)
    cfg.flows.hidden_sizes = [64, 64, 64, 64]
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
    cfg.scheduler.decay_steps = 500_000  # include warmup steps
    cfg.scheduler.warmup_steps = 40_000
    cfg.scheduler.eta_min = 0.01
    cfg.scheduler.interval = 'step'

    # training loop configuration
    cfg.num_steps = 500_000
    cfg.patience = 100
    cfg.monitor = 'val_loss'
    cfg.mode = 'min'
    cfg.grad_clip = 0.5
    cfg.save_top_k = 5
    cfg.accelerator = 'gpu'

    return cfg
