from typing import Generic, TypeVar

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from samplers.dtypes import Shape
from samplers.inverse_problem import InverseProblem
from samplers.samplers.base import PosteriorSampler
from samplers.samplers.utils.batch_view import BatchView
from samplers.samplers.utils.bridge_kernels import ddim_step

Condition_co = TypeVar("Condition_co", covariant=True)


class DPSSampler(PosteriorSampler, Generic[Condition_co]):
    """Diffusion Posterior Sampling for solving inverse problems with diffusion
    models.

    This implementation is based on the paper "Diffusion Posterior
    Sampling for General Noisy Inverse Problems" (Chung et al., 2022).
    """

    def __call__(
        self,
        inverse_problem: InverseProblem,
        num_sampling_steps: int = 50,
        num_reconstructions: int = 1,
        gamma: float = 1.0,
        eta: float = 1.0,
        condition: Condition_co | None = None,
        # initial_noise: Tensor | None = None,  # todo
        # callback_fn: Callable | None = None,  # todo potential function after each iteration to see what is happening
        keep_reconstruction_dim: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        """Run DPS algorithm to solve an inverse problem.

        Args:
            inverse_problem: The inverse problem to solve, containing the observation,
                forward operator, and noise model.
            num_sampling_steps: Number of diffusion timesteps to use for sampling.
                Fewer steps are faster but may produce lower quality results.
            num_reconstructions: Number of independent reconstructions to generate.
                Multiple samples help capture posterior uncertainty.
            gamma: Step size for the likelihood gradient. Controls the strength of the
                measurement consistency correction.
            eta: Parameter for the DDIM step. Values range from 0 (deterministic DDIM)
                to 1 (stochastic, DDPM-like behavior).

        Returns:
            Tensor containing reconstructed samples with shape
            (*batch_shape, num_reconstructions, *x_shape).

        Raises:
            AttributeError: If the epsilon network is missing required attributes.
        """

        if args or kwargs:
            print(f"Warning: Unused args={args}, kwargs={kwargs} in DPSSampler")

        # 1. Setup BatchView for shape management
        x_shape: Shape = inverse_problem.operator.x_shape
        batch_shape: Shape = inverse_problem.batch_shape
        view = BatchView(
            batch_shape=batch_shape,
            num_samples=num_reconstructions,
            data_shape=x_shape,
        )

        # 2. Setup diffusion model and validation
        epsilon_net = self._epsilon_network
        epsilon_net.set_sampling_parameters(
            num_sampling_steps=num_sampling_steps,
            num_reconstructions=num_reconstructions,
            batch_size=view.batch_size,
        )
        epsilon_net.set_condition(condition=condition)

        # 3. Initialize sampling tensor (random noise)
        sample = torch.randn(
            size=view.flat_shape,
            device=epsilon_net.device,
            dtype=epsilon_net.dtype,
        )

        # 4. Sampling loop (reverse diffusion with likelihood guidance)
        timesteps = epsilon_net.timesteps
        for i in trange(len(timesteps) - 1, 1, -1):
            timestep = int(timesteps[i])
            timestep_prev = int(timesteps[i - 1])

            # Enable gradient computation for the current noisy sample
            sample = sample.detach().requires_grad_()

            # Predict the clean data x0 from the current noisy sample
            x0_pred = epsilon_net.predict_x0(sample, timestep)

            # Calculate log-likelihood gradient for measurement consistency
            log_l_val = inverse_problem.log_likelihood(x0_pred).sum()
            grad_pot = torch.autograd.grad(log_l_val, sample)[0]

            # Perform standard diffusion model update step (DDIM)
            sample = ddim_step(
                x=sample.detach(),
                epsilon_net=epsilon_net,
                t=timestep,
                t_prev=timestep_prev,
                eta=eta,
                e_t=x0_pred,
            )

            # Apply measurement consistency correction through gradient ascent
            with torch.no_grad():
                residual = inverse_problem.residual(x0_pred)
                residual = residual.view(view.leading_size, -1)
                # L2‑norm per flattened sample, then reshape for broadcasting
                error_val = residual.norm(dim=1).view(view.per_sample_broadcast_shape)
                scaled_grad = (gamma / (error_val + 1e-9)) * grad_pot
                sample = sample + scaled_grad

        # 5. Final clean data prediction and reshape back
        final_timestep = int(timesteps[1])
        x0_final_flat = epsilon_net.predict_x0(sample, final_timestep)
        x0_final = view.unflatten(x0_final_flat)

        if num_reconstructions == 1 and not keep_reconstruction_dim:
            x0_final = x0_final.squeeze(len(batch_shape))

        return x0_final
