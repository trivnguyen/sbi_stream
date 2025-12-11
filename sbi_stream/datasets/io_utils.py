
import h5py
import numpy as np


def write_dataset(path, data):
    """ Save dataset to HDF5 """
    # get the length of each sample and create a pointer
    sample_len = np.array([len(d) for d in data['phi1']])
    ptr = np.cumsum(sample_len)
    ptr = np.insert(ptr, 0, 0)

    # create the dataset
    with h5py.File(path, 'w') as f:
        for key in data.keys():
            f.create_dataset(
                key, data=np.concatenate(data[key]), compression='gzip')
        f.create_dataset('ptr', data=ptr, compression='gzip')

def read_dataset(path, unpack=True):
    """ Read dataset from HDF5 """
    data = {}
    with h5py.File(path, 'r') as f:
        ptr = f['ptr'][:]
        for key in f.keys():
            if key != 'ptr':
                val = f[key][:]
                if unpack:
                    val = np.split(val, ptr[1:-1])
                data[key] = val
    return data, ptr

