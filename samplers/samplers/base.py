from abc import ABC, abstractmethod

from samplers.dtypes import Shape, Tensor
from samplers.networks import Network


class PosteriorSampler(ABC):
    def __init__(self, network: Network):
        self._epsilon_network = network

    @staticmethod
    def _flatten_leading(
        x: Tensor, *, x_shape: Shape
    ) -> tuple[Tensor, Shape]:  # todo create a container
        """Collapse every axis before `x_shape` into one; return the original
        batch_shape."""
        batch_shape = x.shape[: -len(x_shape)]  # () if no batching
        x_flat = x.reshape(-1, *x_shape)  # 1-D batch for eps-net
        return x_flat, batch_shape

    @staticmethod
    def _unflatten_leading(x_flat: Tensor, *, batch_shape: tuple[int, ...]) -> Tensor:
        """Restore the flattened tensor back to the original batch_shape."""
        return x_flat.reshape(*batch_shape, *x_flat.shape[1:])
