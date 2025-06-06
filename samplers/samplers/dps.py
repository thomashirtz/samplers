from typing import Generic, TypeVar

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from samplers.dtypes import Shape
from samplers.inverse_problem import InverseProblem
from samplers.samplers.base import PosteriorSampler
from samplers.samplers.utilities import ddim_step

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

        # --- 2. Initialize sampling tensors ---
        # Get shape information from the inverse problem
        x_shape: Shape = inverse_problem.operator.x_shape  # Data shape (C,H,W)
        batch_shape: Shape = inverse_problem.batch_shape  # Batch dimensions from observation
        batch_size = int(np.prod(batch_shape))
        leading_shape = (*batch_shape, num_reconstructions)
        flat_batch = int(batch_size * num_reconstructions)

        # --- 1. Setup diffusion model and validation ---
        epsilon_net = self._epsilon_network
        epsilon_net.set_sampling_parameters(
            num_sampling_steps=num_sampling_steps,
            num_reconstructions=num_reconstructions,
            batch_size=batch_size,
        )
        epsilon_net.set_condition(condition=condition)

        # Start from random noise (standard normal distribution)
        sample_shape = (flat_batch, *x_shape)
        sample = torch.randn(
            size=sample_shape,
            device=epsilon_net.device,
            dtype=epsilon_net.dtype,
        )

        # --- 3. Sampling Loop (reverse diffusion process with likelihood guidance) ---
        timesteps = epsilon_net.timesteps
        for i in trange(len(timesteps) - 1, 1, -1):
            timestep = int(timesteps[i])  # Current timestep
            timestep_prev = int(timesteps[i - 1])  # Next timestep (going backward)

            # Enable gradient computation for the current noisy sample
            sample.requires_grad_()

            # Predict the clean data from the current noisy sample
            sample.requires_grad_()
            # Get model prediction of the clean data x_0
            x_0t = epsilon_net.predict_x0(sample, timestep)

            # Calculate log-likelihood gradient for measurement consistency
            # This guides the sample toward solutions consistent with the observation
            log_l_val = inverse_problem.log_likelihood(x_0t).sum()
            grad_pot = torch.autograd.grad(log_l_val, sample)[0]

            # Perform standard diffusion model update step using DDIM
            sample = ddim_step(
                x=sample.detach(),  # Detach to prevent gradient accumulation
                epsilon_net=epsilon_net,
                t=timestep,
                t_prev=timestep_prev,
                eta=eta,
                e_t=x_0t,
            )

            # Apply measurement consistency correction through gradient ascent
            with torch.no_grad():
                # Calculate the residual norm for adaptive step size scaling
                residual = inverse_problem.residual(x_0t)

                # Pick every axis **except** the batch axis (dim 0)
                reduce_axes = tuple(range(1, residual.ndim))  # e.g. (1, 2, 3) for (B, C, H, W)
                # Take the L2-norm over those axes, but keep them as length-1 dims
                error_val = torch.linalg.vector_norm(residual, dim=reduce_axes, keepdim=True)

                # Scale gradient by residual norm (with numerical stability term)
                scaled_grad = (gamma / (error_val + 1e-9)) * grad_pot
                # Update sample with the scaled gradient
                sample = sample + scaled_grad

        # --- 4. Final clean data prediction ---
        # Get the final denoised sample using the diffusion model
        final_timestep = int(timesteps[1])
        x0_final_flat = epsilon_net.predict_x0(x=sample, t=final_timestep)
        x0_final = x0_final_flat.reshape(*leading_shape, *x_shape)
        if num_reconstructions == 1 and not keep_reconstruction_dim:
            x0_final = x0_final.squeeze(len(batch_shape))
        return x0_final
