from dataclasses import dataclass

import torch
from torch import Tensor

from samplers.networks.base import EpsilonNetwork


@dataclass(frozen=True)
class BridgeStatistics:
    mean: Tensor
    std: Tensor


def compute_bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNetwork,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    """S < t < ell."""
    dtype = x_ell.dtype
    alphas_cumprod = epsilon_net.alphas_cumprod

    alpha_cumprod_t = alphas_cumprod[t].to(dtype=torch.float64)
    alpha_cumprod_ell = alphas_cumprod[ell].to(dtype=torch.float64)
    alpha_cumprod_s = alphas_cumprod[s].to(dtype=torch.float64)

    alpha_cum_s_to_t = alpha_cumprod_t / alpha_cumprod_s
    alpha_cum_t_to_ell = alpha_cumprod_ell / alpha_cumprod_t
    alpha_cum_s_to_ell = alpha_cumprod_ell / alpha_cumprod_s
    bridge_std = (
        eta * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell)) ** 0.5
    )
    bridge_coeff_ell = ((1 - alpha_cum_s_to_t - bridge_std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    bridge_coeff_s = (alpha_cum_s_to_t**0.5) - bridge_coeff_ell * (alpha_cum_s_to_ell**0.5)

    bridge_mean = bridge_coeff_ell * x_ell + bridge_coeff_s * x_s

    return BridgeStatistics(
        mean=bridge_mean.to(dtype=dtype),
        std=bridge_std.to(dtype=dtype),
    )


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNetwork,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    statistics = compute_bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return statistics.mean + statistics.std * torch.randn_like(statistics.mean)


def ddim_step(
    x: torch.Tensor,
    epsilon_net: EpsilonNetwork,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor | None = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return sample_bridge_kernel(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def _broadcast_schedule_coeff(coeff: Tensor, x: Tensor) -> Tensor:
    return coeff.to(device=x.device, dtype=x.dtype).view(-1, *([1] * (x.ndim - 1)))


def ddim_step_eps(
    x: Tensor,
    *,
    epsilon_net: EpsilonNetwork,
    t: int,
    t_prev: int,
    eta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """DDIM update in epsilon parameterization (returns ``x_prev``, ``x0``,
    pseudo-``x0``).

    Used by ReSample; DPS/PGDM use :func:`ddim_step` (x₀ bridge formulation).
    """
    acp_t = epsilon_net.alphas_cumprod[t]
    acp_prev = epsilon_net.alphas_cumprod[t_prev]

    with torch.no_grad():
        e_t = epsilon_net.predict_noise(x, t)

    sqrt_oma_t = (1 - acp_t).sqrt()
    a_t = _broadcast_schedule_coeff(acp_t, x)
    a_prev = _broadcast_schedule_coeff(acp_prev, x)
    sqrt_oma = _broadcast_schedule_coeff(sqrt_oma_t, x)

    pred_x0 = (x - sqrt_oma * e_t) / a_t.sqrt()
    pseudo_x0 = (x - (1 - acp_t) * e_t) / a_t.sqrt()

    sigma_t = eta * ((1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)).clamp(min=0).sqrt()
    sigma = _broadcast_schedule_coeff(sigma_t, x)
    dir_xt = (1 - acp_prev - sigma_t**2).clamp(min=0).sqrt() * e_t
    noise = sigma * torch.randn_like(x)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

    return x_prev, pred_x0, pseudo_x0
