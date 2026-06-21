"""ReSample-only helpers (hard consistency optimization, stochastic resample).

Shared diffusion math lives in :mod:`bridge_kernels` (``ddim_step_eps``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from samplers.networks.base import EpsilonNetwork, LatentEpsilonNetwork
from samplers.operators.base import Operator


def dps_conditioning(
    z_leaf: Tensor,
    x_t: Tensor,
    x_0_hat: Tensor,
    measurement: Tensor,
    *,
    operator: Operator,
    epsilon_net: LatentEpsilonNetwork,
    scale: Tensor | float,
) -> Tensor:
    """Posterior-sampling gradient step on ``x_t`` (ReSample DPS hook)."""
    difference = measurement - operator.apply(epsilon_net.decode(x_0_hat, differentiable=True))
    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=z_leaf)[0]
    return x_t - norm_grad * scale


def pixel_optimization(
    measurement: Tensor,
    x_prime: Tensor,
    *,
    operator: Operator,
    eps: float,
    max_iters: int,
) -> Tensor:
    """Argmin :math:`\\|y - A(x)\\|_2^2` in pixel space."""
    loss_fn = torch.nn.MSELoss()
    opt_var = x_prime.detach().clone().requires_grad_()
    optimizer = torch.optim.AdamW([opt_var], lr=1e-2)
    measurement = measurement.detach()

    for _ in range(max_iters):
        optimizer.zero_grad()
        measurement_loss = loss_fn(measurement, operator.apply(opt_var))
        measurement_loss.backward()
        optimizer.step()
        if measurement_loss.item() < eps**2:
            break

    return opt_var.detach()


def latent_optimization(
    measurement: Tensor,
    z_init: Tensor,
    *,
    operator: Operator,
    epsilon_net: LatentEpsilonNetwork,
    eps: float,
    max_iters: int,
) -> Tensor:
    """Argmin :math:`\\|y - A(D(z))\\|_2^2` in latent space."""
    if not z_init.requires_grad:
        z_init = z_init.requires_grad_()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([z_init], lr=5e-3)
    measurement = measurement.detach()
    losses: list[float] = []

    for itr in range(max_iters):
        optimizer.zero_grad()
        decoded = epsilon_net.decode(z_init, differentiable=True)
        output = loss_fn(measurement, operator.apply(decoded))
        output.backward()
        optimizer.step()
        cur_loss = output.detach().item()

        if itr >= 200:
            losses.append(cur_loss)
            if len(losses) > 1 and losses[0] < cur_loss:
                break
            if len(losses) > 1:
                losses.pop(0)

        if cur_loss < eps**2:
            break

    return z_init.detach()


def stochastic_resample(
    pseudo_x0: Tensor,
    x_t: Tensor,
    a_t: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Stochastic resample step from the ReSample paper."""
    device = pseudo_x0.device
    noise = torch.randn_like(pseudo_x0, device=device)
    return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t) / (
        sigma + 1 - a_t
    ) + noise * torch.sqrt(1 / (1 / sigma + 1 / (1 - a_t)))


def alpha_broadcast(
    epsilon_net: EpsilonNetwork,
    t: int,
    *,
    batch_size: int,
    spatial_ndim: int,
) -> Tensor:
    """Return ``alphas_cumprod[t]`` shaped ``(batch_size, 1, …, 1)``."""
    acp = epsilon_net.alphas_cumprod[t].to(device=epsilon_net.device, dtype=epsilon_net.dtype)
    shape = (batch_size,) + (1,) * (spatial_ndim - 1)
    return acp.expand(shape)


def compute_sigma(
    sigma_scale: float,
    a_t: Tensor,
    a_prev: Tensor,
) -> Tensor:
    """Variance schedule for stochastic resample."""
    return sigma_scale * (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
