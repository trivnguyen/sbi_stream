
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

def bin_stream_hilmi24(
    phi1: np.ndarray, phi2: np.ndarray, feat: np.ndarray,
    m_coeff: float, b_coeff: float, num_bins: int,
    phi1_min: float = None, phi1_max: float = None,
):
    """ Bin the stream along the phi1 coordinates and compute the mean and stdv
    of the features in each bin. """

    # divide phi1 and phi2 using the line
    phi2_line = m_coeff * phi1 + b_coeff
    above = phi2 > phi2_line
    below = phi2 <= phi2_line

    phi1_min = phi1_min or phi1.min()
    phi1_max = phi1_max or phi1.max()

    phi1_bcent_ab, feat_mean_ab, feat_stdv_ab, feat_count_ab = bin_stream(
        phi1[above], feat[above], num_bins=num_bins,
        phi1_min=phi1_min, phi1_max=phi1_max
    )
    phi1_bcent_bl, feat_mean_bl, feat_stdv_bl, feat_count_bl = bin_stream(
        phi1[below], feat[below], num_bins=num_bins,
        phi1_min=phi1_min, phi1_max=phi1_max
    )
    phi1_bcent = np.concatenate([phi1_bcent_ab, phi1_bcent_bl])
    feat_mean = np.concatenate([feat_mean_ab, feat_mean_bl])
    feat_stdv = np.concatenate([feat_stdv_ab, feat_stdv_bl])
    feat_count = np.concatenate([feat_count_ab, feat_count_bl])

    return phi1_bcent, feat_mean, feat_stdv, feat_count


# Get simulated Gaia G band magnitudes
def gaia_transform(g, r, i, z, version='dr3'):
    """From Eli Rykoff for FGCM comparisons via Slack:
    https://darkenergysurvey.slack.com/archives/C016J5SDV9T/p1608182260343200

    This complex model behaves about as well as a random forest
    classifier for Gaia DR2.  The magnitude dependence of the
    transformation is huge because of background errors in Gaia DR2.

    The EDR3 transformations have a much smaller r-offset curvature
    due to background issues and should be able to go deeper.

    These transformations are valid for 0 < g - i < 1.5

    /nfs/slac/kipac/fs1/g/des/erykoff/des/y6a1/fgcm/run_v2.1.2/gedr3/gaia_superfitmodel3.py

    Parameters
    ----------
    g, r, i, z : DECam magnitudes in g,r,i,z  (AB system)
    version    : Version of the Gaia catalog to compare against

    Returns
    -------
    Gmag  : Predicted Gaia G-band magnitude based on DECam griz
    """

    magConst = 2.5 / np.log(10.0)
    lambdaStd = np.array([4790.28076172, 6403.26367188, 7802.49755859, 9158.77441406])
    i0Std = np.array([0.16008162, 0.18297842, 0.17169334, 0.1337308])
    i1Std = np.array([2.09808350e-05, -1.22070312e-04, -1.08942389e-04, -8.01086426e-05])
    i10Std = i1Std / i0Std
    fudgeFactors = np.array([0.25, 1.0, 1.0, 0.25])
    fudgeShift   = 0.0 # Additive shift in peak (mag)
    nBands = lambdaStd.size

    #g -= -46 * 1e-3
    #r -= 0   * 1e-3
    #i -= 38  * 1e-3
    #z -= 31  * 1e-3

    trainCat = np.rec.fromarrays([g,r,i,z],names=['g','r','i','z'])
    fluxg = 10**(g/-2.5)
    fluxr = 10**(r/-2.5)
    fluxi = 10**(i/-2.5)
    fluxz = 10**(z/-2.5)

    S = np.zeros((trainCat.size, nBands - 1), dtype='f8')

    S[:, 0] = (-1. / magConst) * (trainCat['r'] - trainCat['g']) / (lambdaStd[1] - lambdaStd[0])
    S[:, 1] = (-1. / magConst) * (trainCat['i'] - trainCat['r']) / (lambdaStd[2] - lambdaStd[1])
    S[:, 2] = (-1. / magConst) * (trainCat['z'] - trainCat['i']) / (lambdaStd[3] - lambdaStd[2])

    fnuPrime = np.zeros((trainCat.size, nBands))
    fnuPrime[:, 0] = S[:, 0] + fudgeFactors[0] * (S[:, 1] + S[:, 0])
    fnuPrime[:, 1] = fudgeFactors[1] * (S[:, 0] + S[:, 1]) / 2.0
    fnuPrime[:, 2] = fudgeFactors[2] * (S[:, 1] + S[:, 2]) / 2.0
    fnuPrime[:, 3] = (
    S[:, 2] +
    fudgeFactors[3] *
    ((lambdaStd[3] - lambdaStd[2]) / (lambdaStd[3] - lambdaStd[1])) *
    (S[:, 2] - S[:, 1])
    )

    if version == 'dr2':
        # DR2 fit parameters
        pars = [1.43223290e+00,1.50877061e+00,8.43173013e-01,-5.99023967e-04,
                4.06188382e-01,3.11181978e-01,2.51002598e-01,1.00000000e-05,
                4.94284725e-03,1.80499806e-03]

    elif version == 'edr3':
        # EDR3: Notice that the last two parameters (describing the
        # r-offset curvature due to background issues) are much smaller.
        pars = [2.61815727e+00,2.69372875e+00,1.45644592e+00,-5.99023051e-04,
                3.97535324e-01,3.15794343e-01,2.55484718e-01,1.00000000e-05,
                8.30152817e-04,-3.57980758e-04]
        fudgeShift = 2.4e-3 # Additive shift in peak (mag)
        
    else:
        raise Exception("Unrecognized Gaia version: %s"%version)

    i10g,i10r,i10i,i10z = pars[0:4]
    kg,kr,ki,kz = pars[4:8]
    r1,r2 = pars[8:10]
    desFlux = (kg * fluxg * (1.0 + fnuPrime[:, 0] * i10g) +
               kr * fluxr * (1.0 + fnuPrime[:, 1] * i10r) +
               ki * fluxi * (1.0 + fnuPrime[:, 2] * i10i) +
               kz * fluxz * (1.0 + fnuPrime[:, 3] * i10z))
    mGDES = -2.5 * np.log10(desFlux)
    rMag = -2.5 * np.log10(fluxr)
    mGDES += r1 * (rMag - 17.0) + r2 * (rMag - 17.0)**2.
    mGDES += fudgeShift # Peak shift
    return mGDES

def get_decam_g_r_i_z(iso_name='Dotter', iso_age=11.5, iso_metallicity=0.00016, 
                      iso_distance_modulus=16.807, iso_stellar_mass=2e4
):
    """Get simulated DECam g, r, i, z magnitudes from the AAU isochrone"""
    
    # Simulated AAU isochrone for g and r bands
    iso_g_r = isochrone.factory(
        name=iso_name,
        age=iso_age,  # Age in Gyr
        metallicity=iso_metallicity,  # Approximate Z value for stream metallicity
        distance_modulus=iso_distance_modulus,  # Average distance modulus
    )
    
    # Simulated AAU isochrone for i and z bands
    iso_i_z = isochrone.factory(
        name=iso_name,
        age=iso_age,  # Age in Gyr
        metallicity=iso_metallicity,  # Approximate Z value for stream metallicity
        distance_modulus=iso_distance_modulus,  # Average distance modulus
        band_1='i',
        band_2='z'
    )

    decam_g, decam_r = iso_g_r.simulate(stellar_mass=iso_stellar_mass)
    decam_i, decam_z = iso_i_z.simulate(stellar_mass=iso_stellar_mass)

    return decam_g, decam_r, decam_i, decam_z

def get_gaia_g(decam_g, decam_r, decam_i, decam_z):
    """Get simulated Gaia G band magnitudes from the AAU isochrone"""

    gaia_g = gaia_transform(decam_g, decam_r, decam_i, decam_z, version='edr3')

    filtered_gaia_g = []
    
    color = decam_g - decam_r + 0.04
    min_color = 0.2
    max_color = max(color)

    min_mag = 14.875
    max_mag = 19.512

    for c, m, j in zip(color, decam_r, gaia_g):
        if min_color <= c <= max_color and min_mag <= m <= max_mag:
            filtered_gaia_g.append(j)
            
    return filtered_gaia_g

def get_pm_uncertainties(gaia_g):
    """Get proper motion uncertainties for the simulated stars"""
    
    # Calculate proper motion uncertainties for Gaia DR3
    pmra_unc, pmdec_unc = proper_motion_uncertainty(gaia_g, release='dr3')

    pmra_err = pmra_unc/1000
    pmdec_err = pmdec_unc/1000
        
    return pmra_err, pmdec_err

def add_uncertainty(
    feat: np.ndarray,
    feat_list: list,
    dist: bool = True,
    v_r: bool = True,
    pm_phi1: bool = True,
    pm_phi2: bool = True
):
    """
    Add uncertainties to the features: distances, radial velocities, proper motions in phi1,
    and proper motions in phi2.

    Args:
        feat: np.ndarray
            Input feature array where each column corresponds to an observable.
        features: list
            List of feature names (e.g., ['phi1', 'phi2', 'pm1', 'pm2', 'vr', 'dist'])
            corresponding to the columns in `feat`.
        dist: bool, optional
            Whether to add uncertainty to distances.
        v_r: bool, optional
            Whether to add uncertainty to radial velocities.
        pm_phi1: bool, optional
            Whether to add uncertainty to proper motions in phi1.
        pm_phi2: bool, optional
            Whether to add uncertainty to proper motions in phi2.

    Returns:
        feat: np.ndarray
            Updated feature array with uncertainties added.
    """

    if pm_phi1 or pm_phi2:
        decam_g, decam_r, decam_i, decam_z = get_decam_g_r_i_z()
        gaia_g = get_gaia_g(decam_g, decam_r, decam_i, decam_z)
        pmra_err, pmdec_err = get_pm_uncertainties(np.array(gaia_g))

    if pm_phi1 and "pm1" in features:
        # Add uncertainty to proper motions in phi1
        pm1_idx = features.index("pm1")
        feat[:, pm1_idx] += np.random.normal(
            0, np.random.choice(pmra_err, size=len(feat[:, pm1_idx]))
        )

    if pm_phi2 and "pm2" in features:
        # Add uncertainty to proper motions in phi2
        pm2_idx = features.index("pm2")
        feat[:, pm2_idx] += np.random.normal(
            0, np.random.choice(pmdec_err, size=len(feat[:, pm2_idx]))
        )

    if v_r and "vr" in features:
        # Add uncertainty to radial velocities
        vr_idx = features.index("vr")
        feat[:, vr_idx] += np.random.normal(
            0, 0.1 * np.abs(feat[:, vr_idx])
        )

    if dist and "dist" in features:
        # Add uncertainty to distances
        dist_idx = features.index("dist")
        feat[:, dist_idx] += np.random.normal(
            0, 0.1 * np.abs(feat[:, dist_idx])
        )

    return feat
