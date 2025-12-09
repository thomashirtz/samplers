from torch import Tensor
from torch.nn import Module


class CustomModel(Module):
    # here define the network with the necessary things to make it work
    alpha_cumprods: Tensor
    timesteps: Tensor

    def forward(self, x: Tensor, t: Tensor) -> Tensor: ...
