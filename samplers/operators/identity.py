import torch

from samplers.dtypes import Device, Shape, Tensor

from .linear import LinearOperator


class IdentityOperator(LinearOperator):
    """If flatten=False:  Y = x                 – identity x = Y –
    transpose/inverse.

    If flatten=True:   Y = x.reshape(..., -1)                – flatten per-sample dims
                        x = Y.reshape(..., *x_shape)        – restore original per-sample dims
    """

    def __init__(self, x_shape: Shape, flatten: bool = False) -> None:
        self.flatten = flatten
        super().__init__(x_shape=x_shape)

    def _infer_y_shape(self, x_shape, device: Device = None):
        if self.flatten:
            return (int(torch.tensor(x_shape).prod().item()),)
        return x_shape

    def apply(self, x: Tensor) -> Tensor:
        """
        Forward: H(x)
        input shape:  (*batch, *x_shape)
        output shape: (*batch, *y_shape)
        """
        if self.flatten:
            batch_dims = x.shape[: -len(self.x_shape)]
            return x.reshape(*batch_dims, *self.y_shape)
        else:
            return x

    def apply_transpose(self, y: Tensor) -> Tensor:
        """
        Adjoint / pseudo-inverse: Hᵀ(y)
        input shape:  (*batch, *y_shape)
        output shape: (*batch, *x_shape)
        """
        if self.flatten:
            batch_dims = y.shape[: -len(self.y_shape)]
            return y.reshape(*batch_dims, *self.x_shape)
        else:
            return y

    apply_pseudo_inverse = apply_transpose
