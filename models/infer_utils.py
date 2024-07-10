
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def sample(
    model, data_loader, num_samples=1, return_labels=True,
    norm_dict=None):
    """ Sampling from a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with `flows` attribute.
    data_loader : torch.utils.data.DataLoader
        Data loader for the dataset.
    num_samples : int, optional
        Number of samples to draw from the model. The default is 1.
    return_labels : bool, optional
        Whether to return the labels. The default is True.
    norm_dict : dict, optional
        Dictionary with normalization parameters. The default is None.
    """
    model.eval()

    samples = []
    labels = []

    loop = tqdm(data_loader, desc='Sampling')
    for batch in loop:
        x = batch[0].to(model.device)
        y = batch[1].to(model.device)
        t = batch[2].to(model.device)
        padding_mask = batch[3].to(model.device)

        flow_context = model(x, t, padding_mask)
        sample = model.flows.sample(num_samples, context=flow_context)
        samples.append(sample.cpu().numpy())
        labels.append(y.cpu().numpy())

    samples = np.concatenate(samples, axis=0)
    labels = np.concatenate(labels, axis=0)

    if norm_dict is not None:
        y_loc = norm_dict['y_loc']
        y_scale = norm_dict['y_scale']
        samples = samples * y_scale + y_loc
        labels = labels * y_scale + y_loc

    if return_labels:
        return samples, labels
    return samples

@torch.no_grad()
def sample_no_labels(
    model, data_loader, num_samples=1, norm_dict=None):
    """ Sampling from a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with `flows` attribute.
    data_loader : torch.utils.data.DataLoader
        Data loader for the dataset.
    num_samples : int, optional
        Number of samples to draw from the model. The default is 1.
    norm_dict : dict, optional
        Dictionary with normalization parameters. The default is None.
    """
    model.eval()

    samples = []

    loop = tqdm(data_loader, desc='Sampling')
    for batch in loop:
        x = batch[0].to(model.device)
        t = batch[1].to(model.device)
        padding_mask = batch[2].to(model.device)

        flow_context = model(x, t, padding_mask)
        sample = model.flows.sample(num_samples, context=flow_context)
        samples.append(sample.cpu().numpy())

    samples = np.concatenate(samples, axis=0)

    if norm_dict is not None:
        y_loc = norm_dict['y_loc']
        y_scale = norm_dict['y_scale']
        samples = samples * y_scale + y_loc

    return samples
