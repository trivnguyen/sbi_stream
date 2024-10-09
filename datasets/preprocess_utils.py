
import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def approximate_arc_length(spline, x_arr):
    y_arr = spline(x_arr)
    p2p = np.sqrt((x_arr[1:] - x_arr[:-1]) ** 2 + (y_arr[1:] - y_arr[:-1]) ** 2)
    arclength = np.concatenate(([0], np.cumsum(p2p)))
    return arclength

def project_onto_univariate_spline(data, spline, x_edges):
    """Computes the 1D projection of data onto a spline."""

    # Compute points along the arc of the curve
    arc_edges = approximate_arc_length(spline, x_edges)

    # Compute the spline points at the x_edges
    curve_points = np.c_[x_edges, spline(x_edges)]

    # Project the data onto the curve
    nN, nF = data.shape
    nL, nF2 = curve_points.shape

    if nF != 2:
        raise ValueError("data must be (N, 2)")
    if nF2 != 2:
        raise ValueError("curve_points must be (N, 2)")

    # curve points
    p1 = curve_points[:-1, :]
    p2 = curve_points[1:, :]
    # vector from one point to next  (nL-1, nF)
    viip1 = np.subtract(p2, p1)
    # square distance from one point to next  (nL-1, nF)
    dp2 = np.sum(np.square(viip1), axis=-1)

    # data minus first point  (nN, nL-1, nF)
    dmi = np.subtract(data[:, None, :], p1[None, :, :])

    # The line extending the segment is parameterized as p1 + t (p2 - p1).  The
    # projection falls where t = [(data-p1) . (p2-p1)] / |p2-p1|^2. tM is the
    # matrix of "t"'s.
    # TODO: maybe replace by spline tangent evaluated at the curve_points
    tM = np.sum((dmi * viip1[None, :, :]), axis=-1) / dp2  # (N, nL-1)

    projected_points = p1[None, :, :] + tM[:, :, None] * viip1[None, :, :]

    # add in the nodes and find all the distances
    # the correct "place" to put the data point is within a
    # projection, unless it outside (by an endpoint)
    # or inside, but on the convex side of a segment junction
    all_points = np.empty((nN, 2 * nL - 1, nF), dtype=float)
    all_points[:, 0::2, :] = curve_points
    all_points[:, 1::2, :] = projected_points
    distances = np.linalg.norm(np.subtract(data[:, None, :], all_points), axis=-1)
    # TODO: better on-sky treatment. This is a small-angle / flat-sky
    # approximation.

    # Detect whether it is in the segment. Nodes are considered in the segment. The end segments are allowed to extend.
    not_in_projection = np.zeros(all_points.shape[:-1], dtype=bool)
    not_in_projection[:, 1 + 2 : -2 : 2] = np.logical_or(
        tM[:, 1:-1] <= 0, tM[:, 1:-1] >= 1
    )
    not_in_projection[:, 1] = tM[:, 1] >= 1  # end segs are 1/2 open
    not_in_projection[:, -2] = tM[:, -1] <= 0

    # make distances for not-in-segment infinity
    distances[not_in_projection] = np.inf

    # Find the best distance
    ind_best_distance = np.argmin(distances, axis=-1)

    idx = ind_best_distance // 2
    arc_projected = arc_edges[idx] + (
        (ind_best_distance % 2)
        * tM[np.arange(len(idx)), idx]
        * (arc_edges[idx + 1] - arc_edges[idx])
    )

    return arc_projected


def bin_stream(
    phi1: np.ndarray, feat: np.ndarray, num_bins: int,
    phi1_min: float = None, phi1_max: float = None
):
    """ Bin the stream along the phi1 coordinates and compute the mean and stdv
    of the features in each bin. """

    phi1_min = phi1_min or phi1.min()
    phi1_max = phi1_max or phi1.max()
    phi1_bins = np.linspace(phi1_min, phi1_max, num_bins + 1)
    phi1_bin_centers = 0.5 * (phi1_bins[1:] + phi1_bins[:-1])

    feat_mean = np.zeros((num_bins, feat.shape[1]))
    feat_stdv = np.zeros((num_bins, feat.shape[1]))
    feat_count = np.zeros((num_bins, 1))

    for i in range(num_bins):
        mask = (phi1 >= phi1_bins[i]) & (phi1 <= phi1_bins[i + 1])
        if mask.sum() <= 1:
            continue
        feat_mean[i] = feat[mask].mean(axis=0)
        feat_stdv[i] = feat[mask].std(axis=0)
        feat_count[i] = mask.sum()

    # TODO: find a better to handle this case
    # remove bins with no data
    mask = (feat_stdv.sum(axis=1) != 0)
    phi1_bin_centers = phi1_bin_centers[mask]
    feat_mean = feat_mean[mask]
    feat_stdv = feat_stdv[mask]
    feat_count = feat_count[mask]

    return phi1_bin_centers, feat_mean, feat_stdv, feat_count


def bin_stream_spline(
    phi1: np.ndarray, phi2: np.ndarray, feat: np.ndarray, num_bins=int,
    num_knots: int = None, phi1_min: float = None, phi1_max: float = None,
    phi2_min: float = None, phi2_max: float = None
):
    """
    Calculate the stream track along the phi1-phi2 coordinates, bin the stream along the
    stream track, and compute the mean, stdv, and count of the features in each bin
    """
    phi1_min = phi1_min or phi1.min()
    phi1_max = phi1_max or phi1.max()
    phi2_min = phi2_min or phi2.min()
    phi2_max = phi2_max or phi2.max()
    num_knots = num_knots or num_bins  # if num knots not given, by default set to bins

    # apply min-max cut on the data
    mask = (phi1_min <= phi1) & (phi1 < phi1_max) & (phi2_min <= phi2) & (phi2 < phi2_max)
    phi1 = phi1[mask]
    phi2 = phi2[mask]
    feat = feat[mask]

    # create the univrate spline and calculate the stream track
    sort = np.argsort(phi1)
    phi1 = phi1[sort]
    phi2 = phi2[sort]
    feat = feat[sort]

    phi1_bins = np.linspace(phi1_min, phi1_max, num_knots + 1)
    phi1_bin_centers = 0.5 * (phi1_bins[1:] + phi1_bins[:-1])
    knot_mask = np.array([], dtype=np.int32)
    for i in range(num_bins):
        mask = (phi1 >= phi1_bins[i]) & (phi1 <= phi1_bins[i + 1])
        if mask.sum() > 1:
            knot_mask = np.append(knot_mask, i)
    knot_mask = knot_mask[1:-1]
    knots = phi1_bin_centers[knot_mask]
    spline = LSQUnivariateSpline(phi1, phi2, knots)
    # project onto the spline
    coord = np.c_[phi1, phi2]
    arc_projected = project_onto_univariate_spline(coord, spline, phi1_bins)

    # normalized arc_projected
    # bin the stream over the stream track and compute the bin statistics'
    arc_min, arc_max = arc_projected.min(), arc_projected.max()
    arc_projected  = (arc_projected - arc_min) / (arc_max - arc_min)
    arc_bins = np.linspace(0, 1, num_bins+1)
    arc_bin_centers = 0.5 * (arc_bins[1:] + arc_bins[:-1])

    feat_mean = np.zeros((num_bins, feat.shape[1]))
    feat_stdv = np.zeros((num_bins, feat.shape[1]))
    feat_count = np.zeros((num_bins, 1))
    for i in range(num_bins):
        mask = (arc_bins[i] <= arc_projected) & (arc_projected < arc_bins[i+1])
        if np.sum(mask) == 0:
            continue
        feat_mean[i] = feat[mask].mean(axis=0)
        feat_stdv[i] = feat[mask].std(axis=0)
        feat_count[i] = np.sum(mask)

    # TODO: find a better to handle this case
    # remove bins with no data
    mask = (feat_count.sum(axis=1) != 0)
    arc_bin_centers = arc_bin_centers[mask]
    feat_mean = feat_mean[mask]
    feat_stdv = feat_stdv[mask]
    feat_count = feat_count[mask]

    return arc_bin_centers, feat_mean, feat_stdv, feat_count


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

def subsample_arrays(arrays: list, subsample_factor: int, unpack=False):
    """ Subsample all arrays in the list. Assuming the arrays have the same length """
    num_sample = len(arrays[0])
    num_subsample = int(np.ceil(num_sample / subsample_factor))
    idx = np.random.choice(num_sample, num_subsample, replace=False)
    arrays = [arr[idx] for arr in arrays]
    # if unpack:
        # return (*arrays,)
    return arrays
