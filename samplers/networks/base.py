from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Network(torch.nn.Module, ABC):  # Network vs EpsilonNetwork vs Denoiser

    def __init__(self, alpha_cumprods: Tensor, timesteps: Tensor):
        super().__init__()
        self.register_buffer("alphas_cumprod", alpha_cumprods)
        self.register_buffer("timesteps", timesteps)
        # todo I am trying to find a way to remove timestep from the initialization of the network
        #  because I want it only when doing sampler.__call__, for now I'll keep it as is.

    @abstractmethod
    def model(self, x: Tensor, t: Tensor) -> Tensor:  # noqa: N802
        """Concrete network implementation (UNet, MLP, …)."""
        raise NotImplementedError

    def forward(self, x: Tensor, t: Tensor):
        return self.model(x=x, t=t)

    def predict_x0(self, x: Tensor, t: Tensor):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, t)) / (acp_t**0.5)

    def score(self, x: Tensor, t: Tensor):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - acp_t) ** 0.5


class LatentNetwork(torch.nn.Module, ABC):

    def decode(self, z: Tensor, differentiable: bool = True):
        if not differentiable:
            with torch.no_grad():
                out = self._decode(z=z)
            return out.detach()
        else:
            return self._decode(z=z)

    @abstractmethod
    def _decode(self, z: Tensor): ...

    def encode(self, data: Tensor, differentiable: bool = True):
        if not differentiable:
            with torch.no_grad():
                out = self._encode(data=data)
            return out.detach()
        else:
            return self._encode(data=data)

    @abstractmethod
    def _encode(self, data: Tensor): ...
