from dataclasses import dataclass

import torch
from torch import Tensor

from samplers.noise import NoiseModel
from samplers.operators import Operator


@dataclass
class InverseProblem:
    operator: Operator
    observation: Tensor
    noise: NoiseModel
    latent_shape: tuple[int, ...]  # always set, never None

    def residual(self, x: Tensor) -> Tensor:
        return self.observation - self.operator(x)

    def log_likelihood(self, x: Tensor) -> Tensor:
        return self.noise.log_prob(self.residual(x))

    def score(self, x: Tensor) -> Tensor:
        return self.noise.score(self.residual(x)) * (-1)  # d/dx of log p

    @classmethod
    def from_observation(cls, obs: Tensor, *, operator: Operator, noise: NoiseModel):
        # 1) if operator doesn’t know its input shape, ask it to deduce it
        if operator.input_shape is None:
            operator.input_shape = operator.infer_input_shape_from_obs(obs)
        # 2) still missing output_shape? we can compute that now
        if operator.output_shape is None:
            operator.output_shape = operator.infer_output_shape(operator.input_shape)

        return cls(
            operator=operator, observation=obs, noise=noise, latent_shape=operator.input_shape
        )

    @classmethod
    def from_clean_data(
        cls,
        x_true: Tensor,
        *,
        operator: Operator,
        noise: NoiseModel,
        rng: torch.Generator | None = None,
    ) -> "InverseProblem":
        """Build an `InverseProblem` by *simulating* the observation: y =
        H(x_true) + ε,   ε ~ noise.sample.

        Parameters
        ----------
        x_true   : clean latent tensor  (B,C,H,W …)  range whatever your model uses
        operator : any `Operator`      (H)
        noise    : any `NoiseModel`    (ε distribution)
        rng      : optional torch.Generator for reproducible noise draws.
        """
        # 1) Make operator remember shapes, if it hasn't already
        if operator.input_shape is None:
            operator.input_shape = x_true.shape[1:]
        if operator.output_shape is None:
            operator.output_shape = operator.infer_output_shape(operator.input_shape)

        # 2) Generate synthetic measurement
        with torch.no_grad():
            y_clean = operator(x_true)
            eps = noise.sample(operator.output_shape)  # ε
            if rng is not None:
                eps = eps.clone().to(x_true.device)
                torch.randn_like(eps, generator=rng, out=eps)  # reproducible
                eps.mul_(noise.sigma if hasattr(noise, "sigma") else 1)
            y_obs = y_clean + eps

        return cls(
            operator=operator,
            observation=y_obs,
            noise=noise,
            latent_shape=operator.input_shape,
        )
