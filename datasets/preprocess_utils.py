
import numpy as np

def bin_stream(phi1: np.ndarray, feat: np.ndarray, num_bins: int):
    """ Bin the stream along the phi1 coordinates and compute the mean and stdv
    of the features in each bin. """

    phi1_min, phi2_max = phi1.min(), phi1.max()
    phi1_bins = np.linspace(phi1_min, phi2_max, num_bins + 1)
    phi1_bin_centers = 0.5 * (phi1_bins[1:] + phi1_bins[:-1])

    feat_mean = np.zeros((num_bins, feat.shape[1]))
    feat_stdv = np.zeros((num_bins, feat.shape[1]))
    for i in range(num_bins):
        mask = (phi1 >= phi1_bins[i]) & (phi1 <= phi1_bins[i + 1])
        if mask.sum() <= 1:
            continue
        feat_mean[i] = feat[mask].mean(axis=0)
        feat_stdv[i] = feat[mask].std(axis=0)

    # TODO: find a better to handle this case
    # remove bins with no data
    mask = (feat_stdv.sum(axis=1) != 0)
    phi1_bin_centers = phi1_bin_centers[mask]
    feat_mean = feat_mean[mask]
    feat_stdv = feat_stdv[mask]

    return phi1_bin_centers, feat_mean, feat_stdv

def pad_and_create_mask(features, max_len=None):
    """ Pad and create Transformer mask. """
    if max_len is None:
        max_len = max([f.shape[0] for f in features])

    # create mask (batch_size, max_len)
    # NOTE: that jax mask is 1 for valid entries and 0 for padded entries
    # this is the opposite of the pytorch mask
    # here we are using the PyTorch mask
    mask = np.zeros((len(features), max_len), dtype=bool)
    for i, f in enumerate(features):
        mask[i, f.shape[0]:] = True

    # zero pad features
    padded_features = np.zeros((len(features), max_len, features[0].shape[1]))
    for i, f in enumerate(features):
        padded_features[i, :f.shape[0]] = f
    return padded_features, mask


def subsample_stream(phi1: np.ndarray, feat: np.ndarray, subsample_factor: int):
    """ Subsample the stream by a factor of `subsample_factor`. """
    # subsample the stream
    num_subsample = int(np.ceil(len(phi1) / subsample_factor))
    idx = np.random.choice(len(phi1), num_subsample, replace=False)
    phi1_subsample = phi1[idx]
    feat_subsample = feat[idx]

    return phi1_subsample, feat_subsample