from typing import Generic, Optional, TypeVar

import torch
from torch import Tensor
from tqdm import trange

from samplers.inverse_problem import InverseProblem
from samplers.networks import LatentNetwork
from samplers.samplers.base import PosteriorSampler
from samplers.samplers.utilities import ddim_step

# Assuming 'Network' and 'LatentNetwork' might be needed for type hints
# and that the actual network passed will conform to the requirements.
# from samplers.network import Network, LatentNetwork


# Condition covariant
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
        if not isinstance(self._epsilon_network, LatentNetwork):
            raise TypeError(
                f"{self.__class__.__name__} requires a latent diffusion model, "
                f"but build_network returned a non-latent network "
                f"({type(self._epsilon_network).__name__})."
            )

    def __call__(
        self,
        inverse_problem: InverseProblem,
        num_sampling_steps: int = 100,
        num_reconstructions: int = 1,
        gamma: float = 1.0,
        omega: float = 0.1,
        eta: float = 1.0,
        decode_output: bool = False,
        condition: Condition_co | None = None,
        *args,
        **kwargs,
    ) -> Tensor:
        """Executes the PSLDSampler.

        Parameters
        ----------
        inverse_problem : InverseProblem
            The definition of the inverse problem, including observation,
            operator, and noise model.
        num_sampling_steps : int, default 100
            The number of timesteps for the diffusion sampling process.
        num_reconstructions : int, default 1
            The number of reconstructions to generate.
        gamma : float, default 1.0
            Stepsize for the gluing constraint (referred to as eta in Algorithm 2).
        omega : float, default 0.1
            Stepsize for the likelihood constraint (referred to as gamma in Algorithm 2).
        eta : float, default 1.0
            The eta parameter for the DDIM step (0 for DDIM, 1 for DDPM-like).
        """
        if args or kwargs:
            print(f"Warning: Unused args={args}, kwargs={kwargs} in PSLDSampler")

        # --- 2. Setup ---
        epsilon_net: LatentNetwork = self._epsilon_network

        # Check for required network attributes and methods
        required_attrs = [
            "set_timesteps",
            "predict_x0",
            "timesteps",
            "encode",
            "decode",
            "get_latent_shape",
            "device",
            "dtype",
        ]
        for attr in required_attrs:
            if not hasattr(epsilon_net, attr):
                raise AttributeError(
                    f"epsilon_net is missing required `{attr}`. "
                    f"PSLDSampler requires a network with both diffusion "
                    f"and latent (encode/decode/latent_shape) capabilities."
                )

        # Check for required operator methods
        if not hasattr(inverse_problem.operator, "apply_transpose"):
            raise AttributeError(
                "InverseProblem.operator must have 'apply_transpose' method for PSLD."
            )

        epsilon_net.set_timesteps(num_sampling_steps)
        epsilon_net.set_condition(condition)
        latent_shape = epsilon_net.get_latent_shape(inverse_problem.operator.x_shape)

        # Initialize latent noise z_T
        shape = (num_reconstructions, *latent_shape)
        z_t = torch.randn(
            size=shape,
            device=epsilon_net.device,
            dtype=epsilon_net.dtype,
        )

        timesteps = epsilon_net.timesteps
        obs = inverse_problem.observation
        operator = inverse_problem.operator

        # Precompute H_transpose(y)
        Ht_obs = operator.apply_transpose(obs)

        # --- 3. Sampling Loop ---
        for i in trange(len(timesteps) - 1, 1, -1):
            t = timesteps[i]
            t_prev = timesteps[i - 1]
            t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device, dtype=torch.long)
            t_prev_tensor = torch.full((z_t.shape[0],), t_prev, device=z_t.device, dtype=torch.long)

            z_t.requires_grad_()

            # Predict z0 using the diffusion model
            z_0t = epsilon_net.predict_x0(z_t, t_tensor)

            # Decode z0 to x0 (image space)
            decoded_z_0t = epsilon_net.decode(z_0t)

            # Apply operator H(x0)
            H_decoded_z_0t = operator.apply(decoded_z_0t)

            # Calculate likelihood error ||y - H(x0)||_2
            ll_error = torch.norm(obs - H_decoded_z_0t)

            # Calculate gluing error ||z0 - E(Ht(y) + x0 - Ht(H(x0)))||_2
            x_eff = Ht_obs + decoded_z_0t - operator.apply_transpose(H_decoded_z_0t)
            encoded_x_eff = epsilon_net.encode(x_eff)
            gluing_error = torch.norm(z_0t - encoded_x_eff)

            # Total error and gradient calculation
            error = omega * ll_error + gamma * gluing_error
            grad = torch.autograd.grad(error, z_t)[0]

            # --- Perform update ---
            with torch.no_grad():
                # DDIM update step (using z_0t as e_t, similar to DPS and original PSLD)
                z_t = ddim_step(
                    x=z_t.detach(),
                    epsilon_net=epsilon_net,
                    t=t_tensor,
                    t_prev=t_prev_tensor,
                    eta=eta,
                    e_t=z_0t,
                )
                # Gradient correction step
                z_t = z_t - grad

        # --- 4. Final Prediction ---
        # Get the final z0 prediction at the second to last timestep
        final_t = torch.full((z_t.shape[0],), timesteps[1], device=z_t.device, dtype=torch.long)
        final_z0 = epsilon_net.predict_x0(z_t, final_t)

        # Decode the final latent prediction to get the image x0

        if decode_output:
            return epsilon_net.decode(final_z0, differentiable=False)
        else:
            return final_z0
