import torch
from torch import Tensor
from tqdm import trange

from samplers.inverse_problem import InverseProblem
from samplers.samplers.base import PosteriorSampler
from samplers.samplers.utilities import ddim_step


class DPSSampler(PosteriorSampler):
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
        # initial_noise: Tensor | None = None,
        gamma: float = 1.0,
        eta: float = 1.0,
        # todo add condition
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

        # --- 1. Setup diffusion model and validation ---
        epsilon_net = self._epsilon_network
        # Verify that the network provides all necessary functionality
        for attr in ("set_timesteps", "predict_x0", "timesteps"):
            if not hasattr(epsilon_net, attr):
                raise AttributeError(f"epsilon_net is missing required `{attr}`")

        # Configure sampling timesteps
        epsilon_net.set_timesteps(num_sampling_steps)

        # --- 2. Initialize sampling tensors ---
        # Get shape information from the inverse problem
        x_shape = inverse_problem.operator.x_shape  # Data shape (C,H,W)
        batch_shape = inverse_problem.batch_shape  # Batch dimensions from observation
        # Create shape for multiple reconstructions
        sample_shape = (*batch_shape, num_reconstructions, *x_shape)

        # Start from random noise (standard normal distribution)
        sample = torch.randn(
            size=sample_shape,
            device=epsilon_net.device,
            dtype=epsilon_net.dtype,
        )

        # Create a shape for broadcasting error values correctly
        shape = (sample.shape[0], *(1,) * len(sample.shape[1:]))
        timesteps = epsilon_net.timesteps

        # --- 3. Sampling Loop (reverse diffusion process with likelihood guidance) ---
        for i in trange(len(timesteps) - 1, 1, -1):
            t = int(timesteps[i])  # Current timestep
            t_prev = int(timesteps[i - 1])  # Next timestep (going backward)

            # Enable gradient computation for the current noisy sample
            sample.requires_grad_()

            # Predict the clean data from the current noisy sample
            # Flatten batch dimensions for the diffusion model
            sample_flatten, leading_shape = self._flatten_leading(sample, x_shape=x_shape)
            sample_flatten.requires_grad_()
            # Get model prediction of the clean data x_0
            x_0t_flatten = epsilon_net.predict_x0(sample_flatten, t)
            # Restore original batch dimensions
            x_0t = self._unflatten_leading(x_0t_flatten, batch_shape=leading_shape)

            # Calculate log-likelihood gradient for measurement consistency
            # This guides the sample toward solutions consistent with the observation
            log_l_val = inverse_problem.log_likelihood(x_0t).sum()
            grad_pot = torch.autograd.grad(log_l_val, sample)[0]

            # Perform standard diffusion model update step using DDIM
            sample = ddim_step(
                x=sample.detach(),  # Detach to prevent gradient accumulation
                epsilon_net=epsilon_net,
                t=t,
                t_prev=t_prev,
                eta=eta,
                e_t=x_0t,  # Using predicted clean data for guidance
            )

            # Apply measurement consistency correction through gradient ascent
            with torch.no_grad():
                # Calculate the residual norm for adaptive step size scaling
                residual = inverse_problem.residual(x_0t)
                error_val = residual.view(sample.shape[0], -1).norm(dim=-1).view(shape)

                # Scale gradient by residual norm (with numerical stability term)
                scaled_grad = (gamma / (error_val + 1e-9)) * grad_pot
                # Update sample with the scaled gradient
                sample = sample + scaled_grad

        # --- 4. Final clean data prediction ---
        # Get the final denoised sample using the diffusion model
        final_t = int(timesteps[1])
        x_0t_flatten, leading_shape = self._flatten_leading(sample, x_shape=x_shape)
        x_hat_flatten = epsilon_net.predict_x0(x_0t_flatten, final_t)
        x_hat = self._unflatten_leading(x_hat_flatten, batch_shape=leading_shape)
        return x_hat
