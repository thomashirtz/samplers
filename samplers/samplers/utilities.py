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
    dtype = x_ell.dtype
    # todo fix the dtype, I think it is a little bit shady right now
    # todo maybe give a dtype_computation and dtype_output if dtype output is none it takes the one of x_ell
    # todo is it really necessary the float64 ?
    # fixme also I tried to avoid having a cumprod f64 in the network to clean it but to see if it is a good idea

    alphas_cumprod_f64 = epsilon_net.alphas_cumprod.double()

    alpha_cum_s_to_t = alphas_cumprod_f64[t] / alphas_cumprod_f64[s]
    alpha_cum_t_to_ell = alphas_cumprod_f64[ell] / alphas_cumprod_f64[t]
    alpha_cum_s_to_ell = alphas_cumprod_f64[ell] / alphas_cumprod_f64[s]
    bridge_std = (
        eta * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell)) ** 0.5
    )
    bridge_coeff_ell = ((1 - alpha_cum_s_to_t - bridge_std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    bridge_coeff_s = (alpha_cum_s_to_t**0.5) - bridge_coeff_ell * (alpha_cum_s_to_ell**0.5)

    bridge_mean = bridge_coeff_ell * x_ell + bridge_coeff_s * x_s
    return bridge_mean.to(dtype=dtype), bridge_std.to(
        dtype=dtype
    )  # todo actually I would like to create a container that is called


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
