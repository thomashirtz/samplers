import torch
from torch import Tensor
from abc import ABC, abstractmethod


class Operator(torch.nn.Module, ABC):
    """
    Generic forward model *A* for inverse problems.

    Mandatory
    ---------
    apply(x)  – forward map  y = A(x)

    Optional
    --------
    apply_transpose(y) – adjoint   x = Aᵀ(y)
    apply_pinv(y)      – pseudo-inverse  x = A⁺(y)

    Notes
    -----
    • Store long-lived tensors as *buffers* so they move with `.to()`.
    """

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Forward map: y = A(x)"""
        pass

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint/transpose map: x = Aᵀ(y)."""
        raise NotImplementedError("Transpose not defined for this operator")

    def apply_pinv(self, y: Tensor) -> Tensor:
        """Pseudo-inverse map: x = A⁺(y)."""
        raise NotImplementedError("Pseudo-inverse not defined for this operator")


class NonlinearOperator(Operator):
    """
    Non-linear degradation operator.

    Required
    --------
    apply(x)  – forward map  y = A(x)

    Optional
    --------
    apply_transpose(y) – a Jacobian-transpose action, if available
    apply_pinv(y)      – a (local) pseudo-inverse, if meaningful

    Use this base when *A* is non-linear; the adjoint or pseudo-inverse may not
    exist in closed form.  Override the optional methods only if your model
    supplies suitable approximations (e.g. automatic-differentiation Jacobian).
    """

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        pass
