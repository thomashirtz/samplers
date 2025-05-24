from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NoiseModel(ABC):
    @abstractmethod
    def log_prob(self, residual: Tensor) -> Tensor: ...

    def score(self, residual: Tensor) -> Tensor:
        residual = residual.detach().requires_grad_()
        logp = self.log_prob(residual)
        (grad,) = torch.autograd.grad(logp.sum(), residual, retain_graph=True)
        return grad  # ∇_x log p(y|x)


class GaussianNoise(NoiseModel):
    def __init__(self, sigma: float):
        self.sigma2 = sigma**2

    def log_prob(self, r: Tensor) -> Tensor:
        return -(r.square().sum(dim=(-1, -2, -3))) / (2 * self.sigma2)


class PoissonNoise(NoiseModel):
    def __init__(self, rate: float):
        self.rate = rate

    def log_prob(self, r: Tensor) -> Tensor:
        # r = observed - λHx   (after suitable scaling)
        return -(r.pow(2) / (self.rate + 1e-3)).sum(dim=(-1, -2, -3))
