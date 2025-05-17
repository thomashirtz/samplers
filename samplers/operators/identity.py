from torch import Tensor

from samplers.operators.linear import LinearOperator


class IdentityOperator(LinearOperator):
    def __init__(self, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self._shape = input_shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)
        # fixme I know this is to be data agnostic, but should we really do that ? where is it hard
        #  to handle if we use the "real" data shape ? I mean, we can flatten the x, do the matrix
        #  multiplications and the convert back to the right shape before returning it.

    def T(self, y: Tensor) -> Tensor:
        return y.reshape((y.shape[0],) + self._shape)
        # fixme this is what CGPT advised me to do, can we discuss about the shape that we should
        #  use ?

    def pinv(self, y: Tensor) -> Tensor:
        return self.T(y)
        # fixme in the documentation say what shape is expected and will be returned
        #  (apply vs apply_pinv)
