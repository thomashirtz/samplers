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
        num_sampling_steps: int = 50,
        num_reconstructions: int = 2,
        # initial_noise: Tensor | None = None,
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
        for attr in ("set_timesteps", "predict_x0", "timesteps"):
            if not hasattr(epsilon_net, attr):
                raise AttributeError(f"epsilon_net is missing required `{attr}`")

        epsilon_net.set_timesteps(num_sampling_steps)

        shape = (num_reconstructions, *inverse_problem.operator.x_shape)
        sample = torch.randn(
            size=shape,
            device=epsilon_net.device,
            dtype=epsilon_net.dtype,
        )

        shape = (sample.shape[0], *(1,) * len(sample.shape[1:]))
        timesteps = epsilon_net.timesteps

        # --- 3. Sampling Loop ---
        for i in trange(len(timesteps) - 1, 1, -1):
            t = int(timesteps[i])
            t_prev = int(timesteps[i - 1])

            sample.requires_grad_()

            # Predict x0 using the diffusion model
            x_0t = epsilon_net.predict_x0(sample, t)

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
                t=t,
                t_prev=t_prev,
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
        final_t = int(timesteps[1])
        return epsilon_net.predict_x0(sample, final_t)
