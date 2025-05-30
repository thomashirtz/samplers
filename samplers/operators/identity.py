import torch

from samplers.dtypes import Device, Shape, Tensor

from .linear import LinearOperator


class IdentityOperator(LinearOperator):
    """Identity operator with optional flattening."""

    def __init__(self, x_shape: Shape, flatten: bool = False) -> None:
        """Initialize the identity operator.

        Args:
            x_shape: Shape of the input tensor.
            flatten: If True, flatten the input tensor in the forward pass
                     and reshape it back in the transpose/inverse pass.
        """
        self.flatten = flatten
        super().__init__(x_shape=x_shape)

    def _infer_y_shape(self, x_shape, device: Device = None):
        """Infer the output shape from the input shape.

        Args:
            x_shape: Shape of the input tensor.
            device: Device on which to place the operator (unused).

        Returns:
            Shape of the output tensor.
        """
        if self.flatten:
            return (int(torch.tensor(x_shape).prod().item()),)
        return x_shape

    def apply(self, x: Tensor) -> Tensor:
        """Forward map: $y = A(x)$

        Args:
            x: Input tensor with shape (*batch, *x_shape).

        Returns:
            If flatten=False: The input tensor unchanged.
            If flatten=True: The input tensor flattened to shape (*batch, -1).
        """
        if self.flatten:
            batch_dims = x.shape[: -len(self.x_shape)]
            return x.reshape(*batch_dims, *self.y_shape)
        else:
            return x

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint/transpose map: $x = A^T(y)$

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            If flatten=False: The input tensor unchanged.
            If flatten=True: The input tensor reshaped to (*batch, *x_shape).
        """
        if self.flatten:
            batch_dims = y.shape[: -len(self.y_shape)]
            return y.reshape(*batch_dims, *self.x_shape)
        else:
            return y

    apply_pseudo_inverse = apply_transpose
