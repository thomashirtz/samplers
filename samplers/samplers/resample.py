from typing import Generic, TypeVar

import torch
from torch import Tensor
from tqdm import trange

from samplers.dtypes import Shape
from samplers.inverse_problem import InverseProblem
from samplers.networks import LatentEpsilonNetwork
from samplers.noise import GaussianNoise
from samplers.samplers.base import PosteriorSampler
from samplers.samplers.utils.batch_view import BatchView
from samplers.samplers.utils.bridge_kernels import ddim_step_eps
from samplers.samplers.utils.resample_kernels import (
    alpha_broadcast,
    compute_sigma,
    dps_conditioning,
    latent_optimization,
    pixel_optimization,
    stochastic_resample,
)

Condition_co = TypeVar("Condition_co", covariant=True)


class ReSampleSampler(PosteriorSampler, Generic[Condition_co]):
    """ReSample: latent diffusion inverse problems via hard data consistency [1].

    Operates in latent space and requires a :class:`~samplers.networks.LatentEpsilonNetwork`.
    Returns pixel-space reconstructions by default (``decode_output=True``), matching PSLD.

    References
    ----------
    .. [1] Song, Bowen, et al. "Solving inverse problems with latent diffusion
           models via hard data consistency." arXiv:2307.08123 (2023).
    """

    def __init__(self, network):
        super().__init__(network)

        if not isinstance(self._epsilon_network, LatentEpsilonNetwork):
            raise TypeError(
                f"{self.__class__.__name__} requires a latent diffusion model, "
                f"but build_network returned a non-latent network "
                f"({type(self._epsilon_network).__name__})."
            )

    def __call__(
        self,
        inverse_problem: InverseProblem,
        *,
        num_sampling_steps: int = 100,
        num_reconstructions: int = 1,
        scale: float = 0.3,
        sigma_scale: float = 40.0,
        max_optimization_iters: int = 2000,
        eta: float = 1.0,
        inter_timesteps: int = 5,
        time_travel_interval: int = 10,
        stage_splits: int = 3,
        decode_output: bool = True,
        condition: Condition_co | None = None,
    ) -> Tensor:
        """Run ReSample and return reconstructions.

        Parameters
        ----------
        inverse_problem
            Forward operator ``H``, observation ``y``, and noise model.
        num_sampling_steps
            Number of diffusion steps *T*.
        num_reconstructions
            Independent posterior samples per observation.
        scale
            Base DPS step size (combined dynamically with ``alphas_cumprod``).
        sigma_scale
            Scales stochastic resample variance.
        max_optimization_iters
            Max AdamW iterations for pixel/latent consistency optimization.
        eta
            DDIM stochasticity (0 = deterministic, 1 = DDPM-like).
        inter_timesteps
            Extra denoising steps during time-travel blocks.
        time_travel_interval
            Run a time-travel block every this many loop indices.
        stage_splits
            Split reverse trajectory into pixel- vs latent-optimization stages.
        decode_output
            If *True*, decode final latents to pixel space before returning.
        condition
            Optional conditioning passed through to ``epsilon_network``.
        """
        x_shape: Shape = inverse_problem.operator.x_shape
        batch_shape: Shape = inverse_problem.batch_shape

        x_view = BatchView(batch_shape, num_reconstructions, x_shape)

        epsilon_net: LatentEpsilonNetwork = self._epsilon_network
        latent_shape: Shape = epsilon_net.get_latent_shape(x_shape)
        z_view = BatchView(batch_shape, num_reconstructions, latent_shape)

        flat_batch_size = z_view.leading_size
        spatial_ndim = len(latent_shape) + 1  # batch + latent dims

        epsilon_net.set_sampling_parameters(
            num_sampling_steps=num_sampling_steps,
            num_reconstructions=num_reconstructions,
            batch_size=x_view.batch_size,
        )
        epsilon_net.set_condition(condition)

        try:
            z_t: Tensor = torch.randn(
                z_view.flat_shape,
                device=epsilon_net.device,
                dtype=epsilon_net.dtype,
            ).requires_grad_()

            timesteps = epsilon_net.timesteps
            observation_flat = x_view.repeat_observation(inverse_problem.observation)
            operator = inverse_problem.operator

            if isinstance(inverse_problem.noise, GaussianNoise):
                eps = float(inverse_problem.noise.sigma.item())
            else:
                eps = 1e-3

            total_steps = len(timesteps) - 1
            index_split = total_steps // stage_splits

            for idx in trange(len(timesteps) - 1, 1, -1):
                timestep = int(timesteps[idx])
                timestep_prev = int(timesteps[idx - 1])

                z_t = z_t.detach().requires_grad_()

                z_next, _z0_pred, pseudo_z0 = ddim_step_eps(
                    z_t,
                    epsilon_net=epsilon_net,
                    t=timestep,
                    t_prev=timestep_prev,
                    eta=eta,
                )

                a_t = alpha_broadcast(
                    epsilon_net, timestep, batch_size=flat_batch_size, spatial_ndim=spatial_ndim
                )
                dps_scale = a_t * 0.5

                z_t = dps_conditioning(
                    z_t,
                    z_next,
                    pseudo_z0,
                    observation_flat,
                    operator=operator,
                    epsilon_net=epsilon_net,
                    scale=dps_scale,
                )

                if (
                    idx <= (total_steps - index_split)
                    and idx > 0
                    and idx % time_travel_interval == 0
                ):
                    x_t_snapshot = z_t.detach().clone()

                    for k in range(idx, max(idx - inter_timesteps, 1), -1):
                        if k <= 1:
                            break
                        t_k = int(timesteps[k])
                        t_prev_k = int(timesteps[k - 1])
                        z_t, _z0_k, pseudo_z0 = ddim_step_eps(
                            z_t,
                            epsilon_net=epsilon_net,
                            t=t_k,
                            t_prev=t_prev_k,
                            eta=eta,
                        )

                    a_prev = alpha_broadcast(
                        epsilon_net,
                        timestep_prev,
                        batch_size=flat_batch_size,
                        spatial_ndim=spatial_ndim,
                    )
                    sigma = compute_sigma(sigma_scale, a_t, a_prev)

                    if idx >= index_split:
                        pseudo_z0 = pseudo_z0.detach()
                        x_pixel = epsilon_net.decode(pseudo_z0, differentiable=False)
                        x_opt = pixel_optimization(
                            observation_flat,
                            x_pixel,
                            operator=operator,
                            eps=eps,
                            max_iters=max_optimization_iters,
                        )
                        z_opt = epsilon_net.encode(x_opt, differentiable=False)
                        z_t = stochastic_resample(z_opt, x_t_snapshot, a_prev, sigma)
                        z_t = z_t.requires_grad_()
                    else:
                        z_opt = latent_optimization(
                            observation_flat,
                            pseudo_z0.detach(),
                            operator=operator,
                            epsilon_net=epsilon_net,
                            eps=eps,
                            max_iters=max_optimization_iters,
                        )
                        z_t = stochastic_resample(z_opt, x_t_snapshot, a_prev, sigma)

            final_z0 = latent_optimization(
                observation_flat,
                z_t.detach(),
                operator=operator,
                epsilon_net=epsilon_net,
                eps=eps,
                max_iters=max_optimization_iters,
            )

            if decode_output:
                x0_output = epsilon_net.decode(final_z0, differentiable=False)
                return x_view.unflatten(x0_output)

            return z_view.unflatten(final_z0)
        finally:
            epsilon_net.clear_condition()
            epsilon_net.clear_sampling_parameters()
