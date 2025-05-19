import torch
from torch import Tensor


def pad_zeros(x: Tensor, target_last_dim: int) -> Tensor:
    """Right-pad the last dimension of *x* with zeros so that x.shape[-1] ==
    target_last_dim."""
    if x.shape[-1] == target_last_dim:
        return x
    if x.shape[-1] > target_last_dim:
        return x[..., :target_last_dim]
    pad_len = target_last_dim - x.shape[-1]
    pad = torch.zeros_like(x[..., :pad_len])
    return torch.cat([x, pad], dim=-1)
