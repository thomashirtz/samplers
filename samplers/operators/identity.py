from samplers.operators.linear import LinearOperator
import torch

class IdentityOperator(LinearOperator):
    """Identity operator: flattens inputs and reshapes back."""
    def __init__(self, input_shape):
        self._shape = input_shape

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map: flatten each sample to row-vector."""
        return x.reshape(x.shape[0], -1)  # todo I know this is to be data agnostic, but should we really do that ? where is it hard to handle if we use the "real" data shape ?

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint map: reshape row-vectors back to original shape."""
        return y.reshape((y.shape[0],) + self._shape) # fixme this is what CGPT advised me to do, can we discuss about the shape that we should use ?

    def apply_pinv(self, y: torch.Tensor) -> torch.Tensor:
        """Pseudo-inverse map: same as transpose for identity."""
        return self.apply_transpose(y)  # fixme in the documentation say what shape is expected and will be returned (apply vs apply_pinv)
