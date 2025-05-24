import torch
from torch import Tensor

from .linear import LinearOperator


class IdentityOperator(LinearOperator):
    def apply(self, x: Tensor) -> Tensor:
        # H(x)
        return x

    def apply_transpose(self, y: Tensor) -> Tensor:
        # Hᵀ(y)
        return y

    # Moore–Penrose pseudo-inverse of I is I.
    apply_pseudo_inverse = apply_transpose


class FlattenIdentityOperator(LinearOperator):
    """Y = x.reshape(batch, -1)                – flattens each sample x =
    y.reshape(batch, *original_shape)   – inverse / adjoint.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        The *per-sample* shape to be flattened and later restored,
        e.g. (3, 64, 64) for RGB images.
    """

    def __init__(self, input_shape: tuple[int, ...]):
        super().__init__()
        self._shape = tuple(input_shape)
        self._flat_dim = int(torch.prod(torch.tensor(self._shape)).item())

    def apply(self, x: Tensor) -> Tensor:
        # H(x) : (B,*shape) → (B,F)
        return x.reshape(x.shape[0], self._flat_dim)

    def apply_transpose(self, y: Tensor) -> Tensor:
        # Hᵀ(y) : (B,F) → (B,*shape)
        return y.reshape((y.shape[0],) + self._shape)

    # For the identity operator the pseudo-inverse equals the transpose.
    apply_pseudo_inverse = apply_transpose
