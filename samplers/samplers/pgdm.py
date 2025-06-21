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


class PGDMSampler(PosteriorSampler, Generic[Condition_co]):
    """Pseudoinverse-Guided Diffusion Models for solving inverse problems.

    This implementation is based on the paper "Pseudoinverse-Guided
    Diffusion Models for Inverse Problems" (Chung et al., 2023).
    https://arxiv.org/pdf/2501.03030v1
    """

    def __call__(
        self,
        inverse_problem: InverseProblem,
        num_sampling_steps: int = 50,
        num_reconstructions: int = 1,
        guidance_weight: float = 1.0,  # Changed from gamma to guidance_weight
        eta: float = 1.0,
        condition: Condition_co | None = None,
        keep_reconstruction_dim: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        """Run PGDM algorithm to solve an inverse problem.

        Args:
            inverse_problem: The inverse problem to solve. Crucially, its `operator`
                must have a `pseudo_inverse` method.
            num_sampling_steps: Number of diffusion timesteps.
            num_reconstructions: Number of independent reconstructions.
            guidance_weight: Step size for the guidance term. Controls the strength
                of the measurement consistency. Corresponds to `grad_term_weight`.
            eta: Parameter for the DDIM step (noise level).

        Returns:
            Tensor containing reconstructed samples.

        Raises:
            AttributeError: If the epsilon network or the operator is missing
                required attributes (like `pseudo_inverse`).
        """
        if not hasattr(inverse_problem.operator, "pseudo_inverse"):
            raise AttributeError(
                "The operator in the inverse_problem must have a 'pseudo_inverse' method for PGDM."
            )

        if args or kwargs:
            print(f"Warning: Unused args={args}, kwargs={kwargs} in PGDMSampler")

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

        # Optional: A better initialization for PGDM is to start from the pseudo-inverse
        # y0_inv = inverse_problem.operator.pseudo_inverse(inverse_problem.observation)
        # last_timestep = int(epsilon_net.timesteps[-1])
        # sample = epsilon_net.q_sample(y0_inv, last_timestep)

        # 4. Sampling loop (reverse diffusion with pseudoinverse guidance)
        timesteps = epsilon_net.timesteps
        for i in trange(len(timesteps) - 1, 1, -1):
            timestep = int(timesteps[i])
            timestep_prev = int(timesteps[i - 1])

            sample = sample.detach().requires_grad_()

            x0_pred = epsilon_net.predict_x0(sample, timestep)

            # Calculate PGDM consistency loss
            with torch.enable_grad():  # Ensure grads are enabled for this part
                y0_inv = inverse_problem.operator.pseudo_inverse(inverse_problem.observation)
                x0_pred_fwd = inverse_problem.operator.forward(x0_pred)
                x0_pred_inv = inverse_problem.operator.pseudo_inverse(x0_pred_fwd)

                # We need the gradient w.r.t x_t, so we compute loss w.r.t. something differentiable
                # that depends on x_t. Using x0_pred works perfectly.
                consistency_loss = (y0_inv - x0_pred_inv).pow(2).sum()

            # Calculate the gradient of this consistency term with respect to the noisy sample x_t
            grad = torch.autograd.grad(consistency_loss, sample)[0]

            # Perform standard DDIM step
            e_t = epsilon_net.predict_noise(sample.detach(), timestep)
            sample_ddim = ddim_step(
                x=sample.detach(),
                epsilon_net=epsilon_net,
                t=timestep,
                t_prev=timestep_prev,
                eta=eta,
                e_t=e_t,
            )

            with torch.no_grad():
                # --- Correct VP Scaling for the Guidance Term ---
                # Get the cumulative alpha for the current timestep t
                alphas_cumprod = epsilon_net.alphas_cumprod
                acp_t = alphas_cumprod[timestep]

                # The scaling factor ensures the gradient step is proportional to the noise level.
                # This is a common and effective scaling for VP models.
                # The gradient was on a loss of x0, but taken w.r.t xt. This scaling is crucial.
                # A factor of sqrt(1 - alpha_t) is theoretically sound for guiding the score.
                scale = guidance_weight * torch.sqrt(1 - acp_t)

                # Apply the guided update. We subtract because we want to descend the loss.
                sample = sample_ddim - scale * grad

        # 5. Final clean data prediction and reshape back
        final_timestep = int(timesteps[1])
        x0_final_flat = epsilon_net.predict_x0(sample, final_timestep)
        x0_final = view.unflatten(x0_final_flat)

        if num_reconstructions == 1 and not keep_reconstruction_dim:
            x0_final = x0_final.squeeze(len(batch_shape))

        epsilon_net.clear_condition()
        epsilon_net.clear_sampling_parameters()
        return x0_final
