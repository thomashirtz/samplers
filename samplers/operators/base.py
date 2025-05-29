from abc import ABC, abstractmethod

import torch

from samplers.dtypes import Device, Shape, Tensor


class Operator(torch.nn.Module, ABC):
    """Generic forward model *A* for inverse problems.

    Mandatory
    ---------
    Operator(x)  – forward map  y = A(x)

    Optional
    --------
    Operator.T(y) – adjoint   x = Aᵀ(y)
    Operator.pinv(y)      – pseudo-inverse  x = A⁺(y)

    Notes
    -----
    • Store long-lived tensors as *buffers* so they move with `.to()`.
    """

    def __init__(self, x_shape: Shape, device: Device = None) -> None:
        super().__init__()
        self.x_shape = tuple(x_shape)  # todo still hesitating between input_shape and x_shape
        self.y_shape = self._infer_y_shape(
            self.x_shape,
            device=device,
        )  # todo still hesitating between input_shape and y_shape

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Forward map: y = A(x)"""
        pass

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint/transpose map: x = Aᵀ(y)."""
        raise NotImplementedError("Transpose not defined for this operator")

    def apply_pseudo_inverse(self, y: Tensor) -> Tensor:
        """Pseudo-inverse map: x = A⁺(y)."""
        raise NotImplementedError("Pseudo-inverse not defined for this operator")

    def forward(self, x: Tensor) -> Tensor:
        return self.apply(x)

    def _infer_y_shape(self, x_shape: Shape, device: Device = None) -> Shape:
        """Run a dummy batch-size-1 through `apply` to get y.shape[1:]."""
        dummy = torch.zeros((1, *x_shape), dtype=torch.float32)
        dummy = dummy.to(next(self.parameters(), torch.tensor([], device=device)).device)
        y = self.apply(dummy)
        return tuple(y.shape[1:])


class NonlinearOperator(Operator):
    """Non-linear degradation operator.

    Required
    --------
    Operator(x)  – forward map  y = A(x)

    Optional
    --------
    Operator.T(y) – a Jacobian-transpose action, if available
    Operator.pinv(y)      – a (local) pseudo-inverse, if meaningful

    Use this base when *A* is non-linear; the adjoint or pseudo-inverse may not
    exist in closed form.  Override the optional methods only if your model
    supplies suitable approximations (e.g. automatic-differentiation Jacobian).
    """

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        pass
