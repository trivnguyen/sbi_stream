
from . import particle_dataset, binned_dataset, matched_filter_dataset

_REGISTRY = {
    'particle': particle_dataset,
    'binned': binned_dataset,
    'matched_filter': matched_filter_dataset,
}


def register_dataset(name, module):
    """Register a new dataset type.

    Parameters
    ----------
    name : str
        The dataset type name used in ``dataset_type`` arguments.
    module : module
        A module exposing ``read_and_process_raw``, ``read_processed``,
        ``prepare_dataloaders``, and ``prepare_test_dataloader``.
    """
    _REGISTRY[name] = module


def _get(dataset_type):
    if dataset_type not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Registered types: {list(_REGISTRY)}.")
    return _REGISTRY[dataset_type]


def read_and_process_raw_datasets(data_dir, dataset_type, **kwargs):
    """Read raw files and process them into a dataset.

    Parameters
    ----------
    data_dir : str or Path
    dataset_type : str
        One of the registered types (e.g. 'particle', 'binned', 'matched_filter').
    **kwargs
        Forwarded to the dataset module's ``read_and_process_raw``.
    """
    return _get(dataset_type).read_and_process_raw(data_dir, **kwargs)


def read_processed_datasets(data_dir, dataset_type, **kwargs):
    """Load preprocessed datasets from pickle files.

    Parameters
    ----------
    data_dir : str or Path
    dataset_type : str
        One of the registered types.
    **kwargs
        Forwarded to the dataset module's ``read_processed``.
    """
    return _get(dataset_type).read_processed(data_dir, **kwargs)


def prepare_dataloaders(data, dataset_type, **kwargs):
    """Create train/val dataloaders.

    Parameters
    ----------
    data : object
        Dataset returned by one of the ``read_*`` functions.
    dataset_type : str
        One of the registered types.
    **kwargs
        Forwarded to the dataset module's ``prepare_dataloaders``.

    Returns
    -------
    tuple
        (train_loader, val_loader, norm_dict)
    """
    return _get(dataset_type).prepare_dataloaders(data, **kwargs)


def prepare_test_dataloader(data, norm_dict, dataset_type, **kwargs):
    """Create a test dataloader.

    Parameters
    ----------
    data : object
    norm_dict : dict
        Normalization parameters from the training run.
    dataset_type : str
        One of the registered types.
    **kwargs
        Forwarded to the dataset module's ``prepare_test_dataloader``.
    """
    return _get(dataset_type).prepare_test_dataloader(data, norm_dict, **kwargs)


__all__ = [
    'register_dataset',
    'read_and_process_raw_datasets',
    'read_processed_datasets',
    'prepare_dataloaders',
    'prepare_test_dataloader',
]
