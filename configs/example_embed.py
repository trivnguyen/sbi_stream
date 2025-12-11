
import optuna
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 42
    config.seed_training = 24

    # data configuration
    config.data = ConfigDict()
    config.data.data_type = 'preprocessed'
    config.data.root = '/mnt/ceph/users/tnguyen/stream/preprocessed_datasets/particles/'
    config.data.name = 'present-6D-sf5'
    config.data.features = ['phi1', 'phi2', 'pm1', 'pm2', 'vr', 'dist']
    config.data.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r', 'phi']
    config.data.num_datasets = 1
    config.data.start_dataset = 0
    config.data.num_subsamples = 5
    config.train_frac = 0.8
    config.num_workers = 0

    ## LOGGING AND WANDB CONFIGURATION ###
    config.workdir = './example'
    config.wandb_project = 'sbi_stream_example_pretrained_embed'
    config.tags = ['embedding', 'debug', 'vmim_loss']
    config.debug = True
    config.checkpoint = None
    config.reset_optimizer = False
    config.enable_progress_bar = True
    config.log_model = 'all'  # Log model checkpoints to WandB


    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = len(config.data.features)
    model.type = 'gnn'  # choices: 'gnn', 'transformer'

    # GNN configuration
    model.gnn = ConfigDict()
    model.gnn.projection_size = 32
    model.gnn.hidden_sizes = [64, ] * 3
    model.gnn.graph_layer = 'ChebConv'
    model.gnn.graph_layer_params = ConfigDict()
    model.gnn.graph_layer_params.sym = True
    model.gnn.graph_layer_params.K = 4
    model.gnn.pooling = 'mean'
    model.gnn.layer_norm = True
    model.gnn.norm_first = False
    model.gnn.act_name = 'leaky_relu'
    model.gnn.act_args = {'negative_slope': 0.01}

    # MLP configuration
    model.mlp = ConfigDict()
    model.mlp.hidden_sizes = [64, ] * 3
    model.mlp.output_size = 5
    model.mlp.dropout = 0.4
    model.mlp.batch_norm = True
    model.mlp.act_name = 'leaky_relu'
    model.mlp.act_args = {'negative_slope': 0.01}

    # Loss configuration
    model.loss_type = 'vmim'  # choices: 'mse', 'vmim' (mse has no args)
    model.loss_args = ConfigDict()
    model.loss_args.features = len(config.data.labels)
    model.loss_args.context_features = 5
    model.loss_args.num_transforms = 6
    model.loss_args.hidden_features = [64, 64]
    model.loss_args.num_bins = 8
    model.loss_args.activation = 'tanh'

    # Pre-transformation configuration
    # this is the transformation that is applied to the data before forward pass
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = True
    pre_transforms.graph_name = 'KNN'
    pre_transforms.graph_args = {'k': 20, 'loop': True}

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 1e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.decay_steps = 10_000
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'

    ### TRAINING CONFIGURATION ###
    config.accelerator = 'gpu'
    config.train_batch_size = 128
    config.eval_batch_size = 128
    config.num_epochs = 20  # example run; set appropriately in real runs
    config.num_steps = -1
    config.patience = 15
    config.gradient_clip_val = 0.5
    config.save_top_k = 5

    return config
