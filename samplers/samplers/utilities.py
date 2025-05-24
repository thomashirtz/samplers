import torch
from torch import Tensor

from samplers.networks.base import Network


def bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: Network,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    """S < t < ell."""
    f8 = torch.float64

    alpha_cum_s_to_t = epsilon_net.acp_f8[t] / epsilon_net.acp_f8[s]
    alpha_cum_t_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[t]
    alpha_cum_s_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[s]
    std = (
        eta * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell)) ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)

    coeff_xell, coeff_xs, std = (
        coeff_xell.to(dtype=f8),
        coeff_xs.to(dtype=f8),
        std.to(dtype=f8),
    )
    return coeff_xell * x_ell + coeff_xs * x_s, std


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: Network,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return mean + std * torch.randn_like(mean)


def ddim_step(
    x: torch.Tensor,
    epsilon_net: Network,
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
