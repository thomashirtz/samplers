from typing import Generic, TypeVar

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from samplers.dtypes import Shape
from samplers.inverse_problem import InverseProblem
from samplers.networks import LatentEpsilonNetwork
from samplers.samplers.base import PosteriorSampler

from .utils.batch_view import BatchView
from .utils.bridge_kernels import ddim_step

Condition_co = TypeVar("Condition_co", covariant=True)


class PSLDSampler(PosteriorSampler, Generic[Condition_co]):
    """Posterior Sampling with Latent Diffusion (PSLD). Implements Algorithm 2
    from [1], adapted to the new library structure.

    This sampler operates in the latent space. It requires that the
    `_epsilon_network` provided to it implements methods from both
    `Network` (like `predict_x0`, `set_timesteps`) and `LatentNetwork`
    (like `encode`, `decode`), and additionally has a `latent_shape`
    attribute.

    References
    ----------
    .. [1] Rout, Litu, et al. "Solving linear inverse problems provably via
           posterior sampling with latent diffusion models."
           Advances in Neural Information Processing Systems 36 (2024).
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
        gamma: float = 1.0,
        omega: float = 0.1,
        eta: float = 1.0,
        decode_output: bool = True,
        condition: Condition_co | None = None,
    ) -> Tensor:
        """Run PSLD and return Monte‑Carlo reconstructions.

        Parameters
        ----------
        inverse_problem
            Specification containing the forward operator ``H`` and
            observation ``y``.
        num_sampling_steps
            Number of diffusion steps *T*.
        num_reconstructions
            Independent posterior samples to draw per observation.
        gamma, omega
            Step sizes for the gluing and likelihood penalties.
        eta
            DDIM stochasticity (0 = deterministic, 1 = DDPM‑like).
        decode_output
            If *True*, return images in observation space.  Otherwise return
            the corresponding latent codes :math:`z_0`.
        condition
            Optional conditioning passed through to ``epsilon_network``.
        """

        # 1.  Shape bookkeeping via two BatchViews
        x_shape: Shape = inverse_problem.operator.x_shape  # (C, H, W)
        batch_shape: Shape = inverse_problem.batch_shape  # (*B,)

        x_view = BatchView(batch_shape, num_reconstructions, x_shape)

        epsilon_net: LatentEpsilonNetwork = self._epsilon_network
        latent_shape: Shape = epsilon_net.get_latent_shape(x_shape)
        z_view = BatchView(batch_shape, num_reconstructions, latent_shape)

        flat_batch_size: int = z_view.leading_size  # |B| · R

        # 2.  Network setup
        # todo maybe put all __call__ in _call and in __call__ you set and clear the parameters and the condition
        epsilon_net.set_sampling_parameters(
            num_sampling_steps=num_sampling_steps,
            num_reconstructions=num_reconstructions,
            batch_size=x_view.batch_size,
        )
        epsilon_net.set_condition(condition)

        # 3.  Initialise latent noise z_T
        z_t: Tensor = torch.randn(
            (flat_batch_size, *latent_shape),
            device=epsilon_net.device,
            dtype=epsilon_net.dtype,
        )

        # 4.  Pre‑compute constants
        timesteps = epsilon_net.timesteps  # list[int]

        observation_flat: Tensor = x_view.repeat_observation(inverse_problem.observation)
        operator = inverse_problem.operator
        H_transpose_observation_flat: Tensor = x_view.repeat_observation(
            operator.apply_transpose(inverse_problem.observation)
        )

        # 5.  Reverse diffusion loop
        for i in trange(len(timesteps) - 1, 1, -1):
            timestep: int = int(timesteps[i])
            timestep_previous: int = int(timesteps[i - 1])

            z_t.requires_grad_()

            # --- 5a.  Network predictions
            z0_prediction: Tensor = epsilon_net.predict_x0(z_t, timestep)
            x0_prediction: Tensor = epsilon_net.decode(z0_prediction, differentiable=True)

            # --- 5b.  Data‑consistency penalties
            H_x0_prediction: Tensor = operator.apply(x0_prediction)
            likelihood_error: Tensor = torch.norm(observation_flat - H_x0_prediction)

            x_effective: Tensor = (
                H_transpose_observation_flat
                + x0_prediction
                - operator.apply_transpose(H_x0_prediction)
            )
            z_effective: Tensor = epsilon_net.encode(x_effective, differentiable=True)
            gluing_error: Tensor = torch.norm(z0_prediction - z_effective)

            total_error: Tensor = omega * likelihood_error + gamma * gluing_error
            (gradient,) = torch.autograd.grad(total_error, z_t)

            # --- 5c.  DDIM step followed by gradient correction
            with torch.no_grad():
                z_t = ddim_step(
                    x=z_t.detach(),
                    epsilon_net=epsilon_net,
                    t=timestep,
                    t_prev=timestep_previous,
                    eta=eta,
                    e_t=z0_prediction,
                )
                z_t = z_t - gradient

        # 6.  Final prediction at timestep 1 (t₁)
        final_z0: Tensor = epsilon_net.predict_x0(z_t, int(timesteps[1]))

        # 7.  Decode and restore structured view
        if decode_output:
            x0_output: Tensor = epsilon_net.decode(final_z0, differentiable=False)
            return x_view.unflatten(x0_output)

        epsilon_net.clear_condition()
        epsilon_net.clear_sampling_parameters()

        return z_view.unflatten(final_z0)
