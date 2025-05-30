from abc import ABC, abstractmethod

import torch

from samplers.dtypes import Device, Shape, Tensor


class Operator(torch.nn.Module, ABC):
    """Generic forward model $A$ for inverse problems.

    Mandatory:
      apply(x) - forward map $y = A(x)$

    Optional:
      apply_transpose(y) - adjoint $x = A^T(y)$
      apply_pseudo_inverse(y) - pseudo-inverse $x = A^+(y)$

    Note:
      Store long-lived tensors as *buffers* so they move with `.to()`.
    """

    def __init__(self, x_shape: Shape, device: Device = None) -> None:
        """Initialize the operator.

        Args:
            x_shape: Shape of the input tensor.
            device: Device on which to place the operator.
        """
        super().__init__()
        self.x_shape = tuple(x_shape)
        self.y_shape = self._infer_y_shape(
            self.x_shape,
            device=device,
        )

    def _infer_y_shape(self, x_shape: Shape, device: Device = None) -> Shape:
        """Run a dummy batch-size-1 through `apply` to get `y_shape`.

        Args:
            x_shape: Shape of the input tensor.
            device: Device on which to place the dummy tensor.

        Returns:
            Shape of an observation (excluding batch dimension).
        """
        device = device or next(self.buffers(), torch.tensor(0)).device
        dummy = torch.zeros((1, *x_shape), dtype=torch.float32, device=device)
        y = self.apply(dummy)
        return tuple(y.shape[1:])

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Forward map: $y = A(x)$

        Args:
            x: Input tensor with shape (*batch, *x_shape).

        Returns:
            Output tensor with shape (*batch, *y_shape).
        """
        pass

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint/transpose map: $x = A^T(y)$.

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            Output tensor with shape (*batch, *x_shape).

        Raises:
            NotImplementedError: If transpose is not defined for this operator.
        """
        raise NotImplementedError("Transpose not defined for this operator")

    def apply_pseudo_inverse(self, y: Tensor) -> Tensor:
        """Pseudo-inverse map: $x = A^+(y)$.

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            Output tensor with shape (*batch, *x_shape).

        Raises:
            NotImplementedError: If pseudo-inverse is not defined for this operator.
        """
        raise NotImplementedError("Pseudo-inverse not defined for this operator")

    def forward(self, x: Tensor) -> Tensor:
        return self.apply(x)


class NonlinearOperator(Operator):
    """Non-linear degradation operator.

    Required:
      apply(x) - forward map $y = A(x)$

    Optional:
      apply_transpose(y) - a Jacobian-transpose action, if available
      apply_pseudo_inverse(y) - a (local) pseudo-inverse, if meaningful

    Note:
      Use this base when $A$ is non-linear; the adjoint or pseudo-inverse may not
      exist in closed form. Override the optional methods only if your model
      supplies suitable approximations (e.g., automatic-differentiation Jacobian).
    """

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        pass
