"""
noise.py  ·  Generic and concrete noise models for inverse problems
-------------------------------------------------------------------

Each noise model is an `nn.Module`, so

* `.to(device, dtype)` migrates all internal buffers
* `state_dict()` / `load_state_dict()` work automatically
* they can be nested inside other modules (samplers, operators, …)

Implemented laws
----------------
• `GaussianNoise(σ)`   – i.i.d. 𝒩(0, σ²)
• `PoissonNoise(λ)`    – ε = k − λ with k ~ Pois(λ)

Public API
----------
`log_prob(r)`   → log pₑ(r)
`score(r)`      → ∇ᵣ log pₑ(r) (autograd fallback)
`sample(shape)` → ε ∼ pₑ

If *dtype* is omitted in `sample`, the tensor follows the dtype of the internal
buffer (σ or λ), preserving mixed-precision semantics.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from samplers.dtypes import RNG, Device, DType, Shape
from samplers.utils.tensor import validate_tensor_is_scalar


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
        device: Device | None = None,
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

    # --------------------------------------------------------------------- #
    # log-likelihood
    # --------------------------------------------------------------------- #
    def log_prob(self, r: Tensor) -> Tensor:
        var = self.sigma.pow(2)
        return -(r.square().sum(dim=tuple(range(1, r.ndim)))) / (2 * var)

    # --------------------------------------------------------------------- #
    # sampling
    # --------------------------------------------------------------------- #
    def sample(
        self,
        shape: Shape,
        *,
        device: Device | None = None,
        dtype: DType = None,
        generator: RNG = None,
    ) -> Tensor:
        device = self.sigma.device if device is None else device
        dtype = self.sigma.dtype if dtype is None else dtype
        eps = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        return eps * self.sigma.to(dtype)


# --------------------------------------------------------------------------- #
# Poisson ε = k − λ,   k ~ Poisson(λ)
# --------------------------------------------------------------------------- #


class PoissonNoise(NoiseModel):
    """Discrete Poisson counting noise.

    Natural residual   ε = k − λ with E[ε]=0, Var[ε]=λ.
    """

    rate: Tensor  # registered buffer

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(self, rate: float | Tensor) -> None:
        """
        Parameters
        ----------
        rate
            Poisson intensity λ (> 0).
        """
        super().__init__()
        rate_tensor = torch.as_tensor(float(rate), dtype=torch.float32)
        if torch.any(rate_tensor <= 0):
            raise ValueError("Poisson rate λ must be positive.")
        self.register_buffer("rate", rate_tensor)

    # --------------------------------------------------------------------- #
    # log-likelihood  (Gaussian approximation form)
    # --------------------------------------------------------------------- #
    def log_prob(self, r: Tensor) -> Tensor:
        return -(r.pow(2) / (self.rate + 1e-3)).sum(dim=tuple(range(1, r.ndim)))

    # --------------------------------------------------------------------- #
    # sampling  – exact Poisson counts centred to zero mean
    # --------------------------------------------------------------------- #
    def sample(
        self,
        shape: Shape,
        *,
        device: Device | None = None,
        dtype: DType = None,
        generator: RNG = None,
    ) -> Tensor:
        device = self.rate.device if device is None else device
        dtype = self.rate.dtype if dtype is None else dtype

        lam = torch.full(shape, self.rate.item(), device=device, dtype=dtype)
        k = torch.poisson(lam, generator=generator)
        return k - lam

        # --- Gaussian approximation (uncomment to use) ------------------ #
        # eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        # return eps * self.rate.sqrt().to(dtype)
