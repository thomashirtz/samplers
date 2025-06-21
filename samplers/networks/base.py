import dataclasses
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

import torch
from torch import Tensor

from samplers.dtypes import Shape

C = TypeVar("C")


class DiffusionType(Enum):
    VARIANCE_PRESERVING = "variance_preserving"
    VARIANCE_EXPLODING = "variance_exploding"
    UNKNOWN = "unknown"


class EpsilonNetwork(torch.nn.Module, ABC, Generic[C]):  # Network vs EpsilonNetwork vs Denoiser
    # todo not sure if I actually like epsilon network, too long, not minimalist enough

    def __init__(self, alphas_cumprod: Tensor):
        super().__init__()
        alphas_cumprod = alphas_cumprod.clip(1e-6, 1)  # todo is it needed to clip ?
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # self.timesteps = None  # todo need to somehow make it abstract property before it is registered
        self._batch_size = None
        self._num_sampling_steps = None
        self._num_reconstructions = None

    @abstractmethod
    def forward(self, x: Tensor, t: Tensor | int): ...

    # todo if it is called Epsilon network, forward can stay as is, however if it is just called network, maybe
    #  it should be called predict noise and predict denoised sample, or somehting like this.

    def predict_noise(self, x: Tensor, t: Tensor | int):
        return self.forward(x, t)

    def predict_x0(self, x: Tensor, t: Tensor | int):
        # todo actually I don't like this naming because it seems like most of the networks are becoming latent,
        #  I would prefer something that is space (pixel space/latent space) agnostic (like sample)
        # predict_denoised_sample
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        # todo it seems unconventional to divide by this value
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, t)) / (acp_t**0.5)

    def score(self, x: Tensor, t: Tensor):  # todo combine forward and score ?
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - acp_t) ** 0.5

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

    @property
    @abstractmethod
    def diffusion_type(self) -> DiffusionType: ...


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
