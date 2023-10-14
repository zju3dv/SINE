def collect_grad(model):
    grad0 = []
    for i, param in enumerate(model.parameters()):
        grad0.append(param.grad.clone().detach())
    return grad0


def collect_param(model):
    param0 = []
    for i, param in enumerate(model.parameters()):
        param0.append(param.clone().detach())
    return param0


import torch
import math


def posenc_window(min_deg, max_deg, alpha):
    """Windows a posenc using a cosiney window.

    This is equivalent to taking a truncated Hann window and sliding it to the
    right along the frequency spectrum.

    Args:
      min_deg: the lower frequency band.
      max_deg: the upper frequency band.
      alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

    Returns:
      A 1-d numpy array with num_sample elements containing the window.
    """
    bands = torch.arange(min_deg, max_deg)
    x = torch.clamp(alpha - bands, 0.0, 1.0)
    return 0.5 * (1 + torch.cos(math.pi * x + math.pi))


def compute_opaqueness_mask(weights, depth_threshold=0.5):
    """Computes a mask which will be 1.0 at the depth point.

    Args:
      weights: the density weights from NeRF.
      depth_threshold: the accumulation threshold which will be used as the depth
        termination point.

    Returns:
      A tensor containing a mask with the same size as weights that has one
        element long the sample dimension that is 1.0. This element is the point
        where the 'surface' is.
    """
    cumulative_contribution = torch.cumsum(weights, axis=-1)
    depth_threshold = torch.tensor(depth_threshold, dtype=weights.dtype)
    opaqueness = cumulative_contribution >= depth_threshold
    false_padding = torch.zeros_like(opaqueness[..., :1])
    padded_opaqueness = torch.cat([false_padding, opaqueness[..., :-1]], axis=-1)
    opaqueness_mask = torch.logical_xor(opaqueness, padded_opaqueness)
    opaqueness_mask = opaqueness_mask.astype(weights.dtype)
    return opaqueness_mask


def compute_depth_index(weights, depth_threshold=0.5):
    """Compute the sample index of the median depth accumulation."""
    opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
    return jnp.argmax(opaqueness_mask, axis=-1)
