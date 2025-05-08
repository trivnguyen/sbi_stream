
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def sample(
    model, data_loader, num_samples=1, return_labels=True, return_log_probs=False,
    norm_dict=None,):
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
    return_log_probs: bool, optional
        Return the log probs of each sample. The default is False.
    norm_dict : dict, optional
        Dictionary with normalization parameters. The default is None.
    """
    model.eval()

    samples = []
    labels = []
    log_probs = []

    loop = tqdm(data_loader, desc='Sampling')
    for batch in loop:
        x = batch[0].to(model.device)
        y = batch[1].to(model.device)
        t = batch[2].to(model.device)
        padding_mask = batch[3].to(model.device)

        flow_context = model(x, t, padding_mask)
        flow = model.flows(flow_context)
        sample = flow.sample((num_samples,))

        if return_log_probs:
            log_prob = flow.log_prob(sample)
            log_prob = torch.transpose(log_prob, 0, 1)
            log_probs.append(log_prob.cpu())

        sample = torch.transpose(sample, 0, 1)  # convert to (batch, num_samples, feat)
        samples.append(sample.cpu())
        labels.append(y.cpu())


    samples = torch.cat(samples, axis=0)
    labels = torch.cat(labels, axis=0)
    if return_log_probs:
        log_probs = torch.cat(log_probs, axis=0)

    if norm_dict is not None:
        y_loc = norm_dict['y_loc']
        y_scale = norm_dict['y_scale']
        if isinstance(y_loc, torch.Tensor):
            y_loc = y_loc.cpu()
        if isinstance(y_scale, torch.Tensor):
            y_scale = y_scale.cpu()

        samples = samples * y_scale + y_loc
        labels = labels * y_scale + y_loc

    return_data = [samples, ]
    if return_labels:
        return_data.append(labels)
    if return_log_probs:
        return_data.append(log_probs)
    return return_data

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
        sample = model.flows(flow_context).sample((num_samples,))
        sample = torch.transpose(sample, 0, 1)  # convert to (batch, num_samples, feat)
        samples.append(sample.cpu().numpy())

    samples = np.concatenate(samples, axis=0)

    if norm_dict is not None:
        y_loc = norm_dict['y_loc']
        y_scale = norm_dict['y_scale'].cpu().numpy()
        samples = samples * y_scale + y_loc

    return samples
