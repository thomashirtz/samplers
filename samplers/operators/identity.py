from torch import Tensor

from samplers.operators.linear import LinearOperator


class IdentityOperator(LinearOperator):
    def __init__(self, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self._shape = input_shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)

    def T(self, y: Tensor) -> Tensor:
        return y.reshape((y.shape[0],) + self._shape)

    def pinv(self, y: Tensor) -> Tensor:
        return self.T(y)
