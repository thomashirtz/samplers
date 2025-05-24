import torch

from samplers.dtypes import Shape, Tensor

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
    y.reshape(batch, *original_shape)  – inverse / adjoint.

    Parameters
    ----------
    x_shape : tuple[int, ...]
        The *per-sample* shape to be flattened and later restored,
        e.g. (3, 64, 64) for RGB images.
    """

    def __init__(self, x_shape: Shape) -> None:
        super().__init__(x_shape=x_shape)
        # total number of features per sample
        self.y_shape = (int(torch.tensor(self.x_shape).prod().item()),)

    def apply(self, x: Tensor) -> Tensor:
        """Flatten the *sample* dimensions of x into one:

        input shape:  (..., *x_shape)
        output shape: (..., flat_dim)
        """
        batch_dims = x.shape[: -len(self.x_shape)]
        return x.reshape(*batch_dims, *self.y_shape)

    def apply_transpose(self, y: Tensor) -> Tensor:
        """
        Restore flattened samples back to x_shape:
          input shape:  (..., flat_dim)
          output shape: (..., *x_shape)
        """
        batch_dims = y.shape[:-1]
        return y.reshape(*batch_dims, *self.x_shape)

    # pseudo-inverse of a flatten is the same as its transpose
    apply_pseudo_inverse = apply_transpose
