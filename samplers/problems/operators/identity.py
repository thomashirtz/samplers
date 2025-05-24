from torch import Tensor

from .linear import LinearOperator


class IdentityOperator(LinearOperator):
    def __init__(self, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self._shape = input_shape

    def apply(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)

    def apply_transpose(self, y: Tensor) -> Tensor:
        return y.reshape((y.shape[0],) + self._shape)

    def apply_pseudo_inverse(self, y: Tensor) -> Tensor:
        return self.apply_transpose(y)
