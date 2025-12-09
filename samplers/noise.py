from abc import ABC, abstractmethod

import torch
from torch import nn

from samplers.dtypes import RNG, Device, DType, Shape, Tensor
from samplers.utils.tensor import validate_tensor_is_scalar

# fixme I did a lot of back and forth with this file, I want it to be a little bit like the reference about how
#  to handle dtype, device etc. Please let me know if there are anything that you don't like.


class NoiseModel(nn.Module, ABC):
    """Abstract noise model with log-probability, score, and sampling."""

    @abstractmethod
    def log_prob(self, residual: Tensor) -> Tensor: ...

    def score(self, residual: Tensor) -> Tensor:
        """∇ log p(residual).

        Override in subclasses if a closed form is cheaper.
        """
        residual = residual.detach().requires_grad_(True)
        logp = self.log_prob(residual).sum()
        (grad,) = torch.autograd.grad(logp, residual, retain_graph=True)
        return grad

    @abstractmethod
    def sample(
        self,
        shape: Shape,
        *,
        device: Device | None = None,
        dtype: DType = None,
        generator: RNG = None,
    ) -> Tensor: ...

    @property
    def device(self) -> torch.device:
        return next(self.buffers()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.buffers()).dtype


class GaussianNoise(NoiseModel):
    """Independent Gaussian noise ε ~ 𝒩(0, σ²)."""

    sigma: Tensor  # registered buffer

    def __init__(
        self,
        sigma: float | Tensor,
        *,
        device: Device = None,
        dtype: DType = None,
    ) -> None:
        super().__init__()

        if isinstance(sigma, Tensor):
            validate_tensor_is_scalar(sigma, "sigma")
            sigma_tensor = sigma.detach().clone()
        else:
            sigma_tensor = torch.tensor(
                float(sigma),
                device=device,
                dtype=dtype or torch.float32,
            )

        if torch.any(sigma_tensor <= 0):
            raise ValueError("σ must be positive.")

        self.register_buffer("sigma", sigma_tensor)

    def log_prob(self, r: Tensor) -> Tensor:
        var = self.sigma.pow(2)
        return -(r.square().sum(dim=tuple(range(1, r.ndim)))) / (2 * var)

    def sample(
        self,
        shape: Shape,
        *,
        device: Device = None,
        dtype: DType = None,
        generator: RNG = None,
    ) -> Tensor:
        device = self.sigma.device if device is None else device
        dtype = self.sigma.dtype if dtype is None else dtype
        eps = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        return eps * self.sigma.to(dtype)


class PoissonNoise(NoiseModel):
    """Discrete Poisson noise: ε = k − λ with k ~ Pois(λ)."""

    rate: Tensor

    def __init__(
        self,
        rate: float | Tensor,
        *,
        device: Device | None = None,
        dtype: DType = None,
    ) -> None:
        super().__init__()
        if isinstance(rate, Tensor):
            validate_tensor_is_scalar(rate, "rate")
            rate_tensor = rate.detach().clone()
        else:
            rate_tensor = torch.tensor(
                float(rate),
                device=device,
                dtype=dtype or torch.float32,
            )
        if rate_tensor <= 0:
            raise ValueError("λ (rate) must be positive.")
        self.register_buffer("rate", rate_tensor)

    def log_prob(self, r: Tensor) -> Tensor:
        # Gaussian-style approximation: Var=λ
        return -(r.pow(2) / (self.rate + 1e-3)).sum(dim=tuple(range(1, r.ndim)))

    def sample(
        self,
        shape: Shape,
        *,
        device: Device = None,
        dtype: DType = None,
        generator: RNG = None,
    ) -> Tensor:
        device = self.rate.device if device is None else device
        dtype = self.rate.dtype if dtype is None else dtype

        lam = torch.full(shape, self.rate.item(), device=device, dtype=dtype)
        k = torch.poisson(lam, generator=generator)
        return k - lam
