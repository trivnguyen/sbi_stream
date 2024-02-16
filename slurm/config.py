
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed = 11

    # data configuration
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = '/mnt/ceph/users/tnguyen/stream_sbi/datasets'
    cfg.data.name = '6params-n1000'
    cfg.data.num_datasets = 1
    cfg.data.num_bins = 20
    cfg.data.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r_sin_phi', 'r_cos_phi']
    cfg.data.label_bounds = config_dict.ConfigDict()
    cfg.data.label_bounds.vtotal = (0, 100)

    # logging configuration
    cfg.workdir = '/mnt/ceph/users/tnguyen/stream_sbi/logging'
    cfg.enable_progress_bar = False
    cfg.overwrite = True

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.8
    cfg.train_batch_size = 1024
    cfg.num_workers = 4
    cfg.eval_batch_size = 1024

    # inference and sampling configuration
    cfg.infer = config_dict.ConfigDict()
    cfg.infer.num_samples = 2000
    cfg.infer.checkpoint = 'best'

    # model configuration
    cfg.output_size = len(cfg.data.labels)
    cfg.featurizer = config_dict.ConfigDict()
    cfg.featurizer.name = 'transformer'
    cfg.featurizer.d_feat_in = 10
    cfg.featurizer.d_time_in = 1
    cfg.featurizer.d_feat = 64
    cfg.featurizer.d_time = 64
    cfg.featurizer.nhead = 4
    cfg.featurizer.num_encoder_layers = 3
    cfg.featurizer.dim_feedforward = 256
    cfg.featurizer.batch_first = True
    cfg.featurizer.activation = config_dict.ConfigDict()
    cfg.featurizer.activation.name = 'Identity'
    cfg.flows = config_dict.ConfigDict()
    cfg.flows.name = 'maf'
    cfg.flows.hidden_size = 128
    cfg.flows.num_blocks = 2
    cfg.flows.num_layers = 4
    cfg.flows.activation = config_dict.ConfigDict()
    cfg.flows.activation.name = 'tanh'

    # optimizer and scheduler configuration
    cfg.optimizer = config_dict.ConfigDict()
    cfg.optimizer.name = 'Adam'
    cfg.optimizer.lr = 3e-4
    cfg.optimizer.betas = (0.9, 0.98)
    cfg.optimizer.weight_decay = 1e-4
    cfg.optimizer.eps = 1e-9
    cfg.scheduler = config_dict.ConfigDict()
    cfg.scheduler.name = 'WarmUpCosineAnnealingLR'
    cfg.scheduler.decay_steps = 100_000  # include warmup steps
    cfg.scheduler.warmup_steps = 5000
    cfg.scheduler.eta_min = 1e-6
    cfg.scheduler.interval = 'step'

    # training loop configuration
    cfg.num_epochs = 1000
    cfg.patience = 100
    cfg.monitor = 'val_loss'
    cfg.mode = 'min'
    cfg.grad_clip = 0.5
    cfg.save_top_k = 5
    cfg.accelerator = 'gpu'

    return cfg
