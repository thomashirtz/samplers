import torch
from torch import Tensor
from abc import ABC, abstractmethod


class Operator(torch.nn.Module, ABC):
    """
    Abstract base for any operator mapping inputs to outputs.
    """
    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Forward map: y = A(x)"""
        pass

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint/transpose map: x = A^T(y)."""
        raise NotImplementedError("Transpose not defined for this operator")

    def apply_pinv(self, y: Tensor) -> Tensor:
        """Pseudo-inverse map: x = A^+(y)."""
        raise NotImplementedError("Pseudo-inverse not defined for this operator")


class NonlinearOperator(Operator):
    """
    Base for non-linear operators. Only apply() required.
    Pseudo-inverse and transpose are optional.
    """
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        pass
