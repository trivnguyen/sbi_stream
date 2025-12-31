"""Configuration file for training NPE on particle datasets."""

from ml_collections import ConfigDict


def get_config():
    """Get the default configuration for training NPE on particle data."""

    config = ConfigDict()
    config.root = '/mnt/ceph/users/tnguyen/stream/'
    config.name = '6params-uni-ta25'
    config.root_out = '/mnt/ceph/users/tnguyen/stream/preprocessed_datasets/particles'
    config.name_out = 'present-aau-6D'
    config.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r', 'phi']
    config.features = ['phi1', 'phi2', 'pm1', 'pm2', 'vr', 'dist']
    config.num_datasets = 50
    config.init = 0
    config.num_subsamples = 1
    config.num_per_subsample = 100
    config.phi1_min = -20
    config.phi1_max = 10
    config.uncertainty_model = 'aau'
    config.include_uncertainty = True
    return config