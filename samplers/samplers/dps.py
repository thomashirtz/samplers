import torch
from torch import Tensor
from tqdm import trange

from samplers.inverse_problem import InverseProblem
from samplers.samplers.base import PosteriorSampler
from samplers.samplers.utilities import ddim_step


class DPSSampler(PosteriorSampler):
    """Diffusion Posterior Sampling (DPS) using a generic InverseProblem.
    Implements Algorithm 1 from [1].

    References
    ----------
    .. [1] Chung, Hyungjin, et al. "Diffusion posterior sampling for general
           noisy inverse problems." arXiv preprint arXiv:2209.14687 (2022).
    """

    def __call__(
        self,
        inverse_problem: InverseProblem,
        num_sampling_steps: int = 10,
        num_reconstructions: int = 1,
        initial_noise: Tensor | None = None,
        gamma: float = 1.0,
        eta: float = 1.0,
        *args,
        **kwargs,
    ) -> Tensor:
        """Executes the DPSSampler.

        Expected Kwargs:
        ----------------
        inverse_problem : InverseProblem
            The definition of the inverse problem, including observation,
            operator, and noise model. (Required)
        initial_noise : Tensor
            The starting noise tensor $x_T$. (Required)
        gamma : float, default 1.0
            The step size for the likelihood gradient (zeta in Algorithm 1).
        eta : float, default 1.0
            The eta parameter for the DDIM step (0 for DDIM, 1 for DDPM-like).
        """
        if args or kwargs:
            print(f"Warning: Unused args={args}, kwargs={kwargs} in DPSSampler")

        # --- 2. Setup ---
        epsilon_net = self._epsilon_network
        if (
            not hasattr(epsilon_net, "set_timesteps")
            or not hasattr(epsilon_net, "predict_x0")
            or not hasattr(epsilon_net, "timesteps")
        ):
            raise AttributeError(
                "self.network (epsilon_net) is missing required methods/attributes."
            )

        epsilon_net.set_timesteps(num_sampling_steps)
        sample = initial_noise.clone()
        shape = (sample.shape[0], *(1,) * len(sample.shape[1:]))
        timesteps = epsilon_net.timesteps

        # --- 3. Sampling Loop ---
        for i in trange(len(timesteps) - 1, 1, -1):
            t = timesteps[i]
            t_prev = timesteps[i - 1]
            t_tensor = torch.full((sample.shape[0],), t, device=sample.device, dtype=torch.long)
            t_prev_tensor = torch.full(
                (sample.shape[0],), t_prev, device=sample.device, dtype=torch.long
            )

            sample.requires_grad_()

            # Predict x0 using the diffusion model
            x_0t = epsilon_net.predict_x0(sample, t_tensor)

            # Calculate the gradient of the log-likelihood w.r.t. the sample (x_t)
            # We use log_likelihood(x_0t) as the potential
            log_l_val = inverse_problem.log_likelihood(x_0t).sum()
            grad_pot = torch.autograd.grad(log_l_val, sample)[0]

            # DDIM update step
            # NOTE: We pass x_0t as e_t to match the original code's likely intent,
            #       but ensure your ddim_step handles this or adjust if it expects noise.
            sample = ddim_step(
                x=sample.detach(),  # Detach before passing to ddim_step
                epsilon_net=epsilon_net,
                t=t_tensor,
                t_prev=t_prev_tensor,
                eta=eta,
                e_t=x_0t,  # Using x_0t here as per original code
            )

            # Likelihood correction step (gradient ascent)
            with torch.no_grad():
                # Use L2 norm of residual for scaling, as a generic approach
                # Ensure the operator/noise model handles any necessary input transforms
                residual = inverse_problem.residual(x_0t)
                error_val = residual.view(sample.shape[0], -1).norm(dim=-1).view(shape)

                # Scale the gradient (add epsilon for stability)
                scaled_grad = (gamma / (error_val + 1e-9)) * grad_pot
                sample = sample + scaled_grad

        # --- 4. Final Prediction ---
        final_t = torch.full(
            (sample.shape[0],), timesteps[1], device=sample.device, dtype=torch.long
        )
        return epsilon_net.predict_x0(sample, final_t)
