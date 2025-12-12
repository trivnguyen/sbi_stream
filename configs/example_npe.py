
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
    config.data.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/graph_npe/datasets/'
    config.data.name = 'present-6D'
    config.data.features = ['phi1', 'phi2', 'pm1', 'pm2', 'vr', 'dist']
    config.data.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r', 'phi']
    config.data.num_datasets = 1
    config.data.start_dataset = 0
    config.data.num_subsamples = 5
    config.train_frac = 0.8
    config.num_workers = 0

    ## LOGGING AND WANDB CONFIGURATION ###
    config.workdir = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/graph_npe'
    config.wandb_project = 'sbi_stream_example'
    config.tags = ['npe', 'debug']
    config.entity = 'desc_sbi_stream'
    config.debug = True
    config.checkpoint = None  # Path to NPE checkpoint for resuming
    config.reset_optimizer = False
    config.enable_progress_bar = True
    config.log_model = 'all'  # Log model checkpoints to WandB

    # Reuse norm_dict from embedding checkpoint
    # if import embedding_nn from a pre-trained model, this should always be True
    config.reuse_embedding_norm_dict = False

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = len(config.data.features)
    model.output_size = len(config.data.labels)

    # Embedding network configuration
    model.embedding = ConfigDict()
    model.embedding.gnn = ConfigDict()
    model.embedding.gnn.projection_size = 32
    model.embedding.gnn.hidden_sizes = [64, ] * 3
    model.embedding.gnn.graph_layer = 'ChebConv'
    model.embedding.gnn.graph_layer_params = ConfigDict()
    model.embedding.gnn.graph_layer_params.K = 4
    model.embedding.gnn.graph_layer_params.sym = True
    model.embedding.gnn.pooling = 'mean'
    model.embedding.gnn.layer_norm = True
    model.embedding.gnn.norm_first = False
    model.embedding.gnn.act_name = 'leaky_relu'
    model.embedding.gnn.act_args = {'negative_slope': 0.01}

    # MLP configuration
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [64, ] * 3
    model.embedding.mlp.output_size = 5
    model.embedding.mlp.dropout = 0.4
    model.embedding.mlp.batch_norm = True
    model.embedding.mlp.act_name = 'leaky_relu'
    model.embedding.mlp.act_args = {'negative_slope': 0.01}

    # NPE Normalizing Flows configuration
    model.flows = ConfigDict()
    model.flows.num_transforms = 6
    model.flows.hidden_features = [64, 64]
    model.flows.num_bins = 8
    model.flows.activation = 'tanh'
    model.flows.randperm = True

    # Pre-transformation configuration
    # Note: For NPE, pre_transforms are passed to NPE, not to embedding_nn
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = True
    pre_transforms.graph_name = 'KNN'
    pre_transforms.graph_args = {'k': 20, 'loop': True}

    ### VISUALIZATION CALLBACK CONFIGURATION ###
    config.enable_visualization_callback = True
    config.visualization = visualization = ConfigDict()
    visualization.n_posterior_samples = 500
    visualization.n_val_samples = 1000
    visualization.plot_every_n_epochs = 1
    visualization.plot_tarp = True
    visualization.plot_median_v_true = True
    visualization.plot_rank = True

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = None
    # scheduler.name = "WarmUpCosineAnnealingLR"
    # scheduler.decay_steps = 10_000
    # scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    # scheduler.eta_min = 1e-6
    # scheduler.interval = 'step'

    ### TRAINING CONFIGURATION ###
    config.accelerator = 'gpu'
    config.train_batch_size = 128
    config.eval_batch_size = 128
    config.num_epochs = 10
    config.num_steps = -1
    config.patience = 15
    config.gradient_clip_val = 0.5
    config.save_top_k = 5

    return config
