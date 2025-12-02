
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed = 103123

    cfg.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/datasets/'
    cfg.name = 'aau-ta25'
    cfg.root_out = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/preprocessed_datasets/AAU'
    cfg.name_out = '2d-AAU-spline-future'

    # data configuration
    cfg.num_datasets = 1
    cfg.start_dataset = 0
    cfg.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r', 'phi']
    cfg.features = ['phi1', 'phi2']
    cfg.subsample_factor = 2.5
    cfg.num_subsamples = 100
    cfg.phi1_min = -20
    cfg.phi1_max = 12
    cfg.use_width = True
    cfg.use_density = False
    cfg.uncertainty = 'future'
    cfg.binning_fn = 'bin_stream_spline'
    cfg.binning_args = binning = config_dict.ConfigDict()
    binning.num_bins = 50
    binning.num_knots = 50
    binning.phi1_min = -20
    binning.phi1_max = 12

    return cfg
