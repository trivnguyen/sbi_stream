def read_raw_dataset(
    data_dir: Union[str, Path], labels: List[str],
    phi1_min: float = None, phi1_max: float = None,
    num_datasets: int = 1
):
    """ Read raw data

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the stream data.
    labels : list of str
        List of labels to use for the regression.
    phi1_min : float, optional
        Minimum value of phi1 to use. Default is None.
    phi1_max : float, optional
        Maximum value of phi1 to use. Default is None.
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    """

    raw = []

    for i in range(num_datasets):
        label_fn = os.path.join(data_dir, f'labels.{i}.csv')
        data_fn = os.path.join(data_dir, f'data.{i}.hdf5')

        if os.path.exists(label_fn) & os.path.exists(data_fn):
            print('Reading in data from {}'.format(data_fn))
        else:
            print('Dataset {} not found. Skipping...'.format(i))
            continue

        # read in the data and label
        table = pd.read_csv(label_fn)
        data, ptr = io_utils.read_dataset(data_fn, unpack=True)

        # compute some derived labels
        table = calculate_derived_properties(table)

        loop = tqdm(range(len(table)))

        for pid in loop:
            loop.set_description(f'Processing pid {pid}')
            phi1 = data['phi1'][pid]
            phi2 = data['phi2'][pid]
            pm1 = data['pm1'][pid]
            pm2 = data['pm2'][pid]
            vr = data['vr'][pid]
            dist = data['dist'][pid]

            raw.append(np.stack([phi1, phi2, pm1, pm2, vr, dist]))

    return raw