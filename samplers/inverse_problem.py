from dataclasses import dataclass

import torch

from samplers.dtypes import RNG, Shape, Tensor
from samplers.noise import NoiseModel
from samplers.operators import Operator


@dataclass
class InverseProblem:
    operator: Operator
    observation: Tensor
    noise: NoiseModel
    # latent_shape: tuple[int, ...]  # always set, never None

    def residual(self, x: Tensor) -> Tensor:
        return self.observation - self.operator(x)

    def log_likelihood(self, x: Tensor) -> Tensor:
        return self.noise.log_prob(self.residual(x))

    def score(self, x: Tensor) -> Tensor:
        return self.noise.score(self.residual(x)) * (-1)  # d/dx of log p

    @property
    def batch_shape(self) -> Shape:
        return self.observation.shape[: -len(self.operator.y_shape)]

    @classmethod
    def from_observation(cls, obs: Tensor, *, operator: Operator, noise: NoiseModel):
        return cls(operator=operator, observation=obs, noise=noise)

    @classmethod
    def from_clean_data(
        cls,
        x_true: Tensor,
        *,
        operator: Operator,
        noise: NoiseModel,
        rng: RNG = None,
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
        with torch.no_grad():
            y_clean = operator(x_true)
            eps = noise.sample(
                shape=y_clean.shape,
                device=y_clean.device,
                dtype=y_clean.dtype,
                generator=rng,
            )
            y_obs = y_clean + eps

        return cls(
            operator=operator,
            observation=y_obs,
            noise=noise,
        )
