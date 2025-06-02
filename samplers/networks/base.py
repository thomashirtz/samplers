import dataclasses
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch import Tensor

from samplers.dtypes import Shape

C = TypeVar("C")


class Network(torch.nn.Module, ABC, Generic[C]):  # Network vs EpsilonNetwork vs Denoiser

    def __init__(self, alphas_cumprod: Tensor):
        super().__init__()
        alphas_cumprod = alphas_cumprod.clip(1e-6, 1)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        # double-precision schedule (for numerically sensitive stats)
        # self.register_buffer("alphas_cumprod_f64", alphas_cumprod.clone().double())
        # it was used in samplers.samplers.utilities.bridge_kernel_statistics, but I would prefer to keep network lean.
        self.register_buffer("timesteps", tensor=None, persistent=False)

        self._batch_size = None
        self._num_sampling_steps = None
        self._num_reconstructions = None

    @abstractmethod
    def forward(self, x: Tensor, t: Tensor | int): ...

    def predict_x0(self, x: Tensor, t: Tensor | int):
        # todo actually I don't like this naming because it seems like most of the networks are becoming latent,
        #  I would prefer something that is space (pixel space/latent space) agnostic (like sample)
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, t)) / (acp_t**0.5)

    def score(self, x: Tensor, t: Tensor):
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
    ):
        self._batch_size = batch_size
        self._num_sampling_steps = num_sampling_steps
        self._num_reconstructions = num_reconstructions

    def clear_sampling_parameters(self):
        self._batch_size = None
        self._num_sampling_steps = None
        self._num_reconstructions = None

    def set_condition(self, condition: C | None) -> None: ...

    def clear_condition(self): ...


class LatentNetwork(Network):

    @abstractmethod
    def get_latent_shape(self, x_shape: Shape) -> Shape: ...

    def decode(self, z: Tensor, differentiable: bool = True):
        if not differentiable:
            with torch.no_grad():
                out = self._decode(z=z)
            return out.detach()
        else:
            return self._decode(z=z)

    @abstractmethod
    def _decode(self, z: Tensor): ...

    def encode(self, x: Tensor, differentiable: bool = True):
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
    prompt/conditioning.”"""

    pass
