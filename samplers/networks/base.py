import dataclasses
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch import Tensor

from samplers.dtypes import Shape

C = TypeVar("C")


class EpsilonNetwork(torch.nn.Module, ABC, Generic[C]):

    """Variance-preserving diffusion prior wrapper.

    Indexing contract (used by samplers and ``bridge_kernels``):

    - ``alphas_cumprod`` is padded: index ``0`` → ``1.0``, index ``k`` → ``alpha_bar_k``.
    - ``timesteps`` buffer is **ascending** (low → high noise index).
    - ``t`` in ``forward`` / ``predict_x0`` / ``ddim_step`` is a buffer index.
    """

    def __init__(self, alphas_cumprod: Tensor):
        super().__init__()
        alphas_cumprod = alphas_cumprod.clip(1e-6, 1)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self._batch_size = None
        self._num_sampling_steps = None
        self._num_reconstructions = None

    @abstractmethod
    def forward(self, x: Tensor, t: Tensor | int): ...

    # todo if it is called Epsilon network, forward can stay as is, however if it is just called network, maybe
    #  it should be called predict noise and predict denoised sample, or something like this.

    def predict_noise(self, x: Tensor, t: Tensor | int):
        return self.forward(x, t)

    def predict_x0(self, x: Tensor, t: Tensor | int):
        acp_t = self.alphas_cumprod[t]
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, t)) / (acp_t**0.5)

    def score(self, x: Tensor, t: Tensor):
        acp_t = self.alphas_cumprod[t]
        return -self.forward(x, t) / ((1 - acp_t) ** 0.5)

    @property
    def device(self) -> torch.device:
        return self.alphas_cumprod.device

    @property
    def dtype(self) -> torch.dtype:
        return self.alphas_cumprod.dtype

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs): ...

    @abstractmethod
    def set_sampling_parameters(
        self,
        num_sampling_steps: int,
        batch_size: int = 1,
        num_reconstructions: int = 1,
    ): ...

    @property
    def are_sampling_parameters_initialized(self) -> bool:
        """True if `set_sampling_parameters()` has been called."""
        return self._batch_size is not None

    def clear_sampling_parameters(self):
        self._batch_size = None
        self._num_sampling_steps = None
        self._num_reconstructions = None

    def set_condition(self, condition: C | None) -> None: ...

    @property
    @abstractmethod
    def is_condition_initialized(self) -> bool: ...

    def clear_condition(self): ...


class LatentEpsilonNetwork(EpsilonNetwork[C], ABC, Generic[C]):

    @abstractmethod
    def get_latent_shape(self, x_shape: Shape) -> Shape: ...

    def decode(self, z: Tensor, differentiable: bool = False):
        if not differentiable:
            with torch.no_grad():
                out = self._decode(z=z)
            return out.detach()
        else:
            return self._decode(z=z)

    @abstractmethod
    def _decode(self, z: Tensor): ...

    def encode(self, x: Tensor, differentiable: bool = False):
        if not differentiable:
            with torch.no_grad():
                out = self._encode(x=x)
            return out.detach()
        else:
            return self._encode(x=x)

    @abstractmethod
    def _encode(self, x: Tensor): ...


@dataclasses.dataclass(slots=True)
class NoCondition:
    """Placeholder type meaning “this diffusion model does NOT use any
    conditioning.”"""
