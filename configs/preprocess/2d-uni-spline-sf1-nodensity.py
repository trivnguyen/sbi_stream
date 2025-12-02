
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed = 65782

    cfg.root = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/datasets/'
    cfg.name = '6params-uni-ta25'
    cfg.root_out = '/pscratch/sd/t/tvnguyen/stream_sbi_shared/npe/preprocessed_datasets/'
    cfg.name_out = '2d-uni-spline-sf1'

    # data configuration
    cfg.num_datasets = 50
    cfg.start_dataset = 0
    cfg.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r', 'phi']
    cfg.features = ['phi1', 'phi2']
    cfg.subsample_factor = 1
    cfg.num_subsamples = 1
    cfg.use_width = True
    cfg.use_density = False
    cfg.binning_fn = 'bin_stream_spline'
    cfg.binning_args = binning = config_dict.ConfigDict()
    binning.num_bins = 50
    binning.num_knots = 50
    binning.phi1_min = -20
    binning.phi1_max = 12

    return cfg
