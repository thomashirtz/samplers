from abc import abstractmethod
from samplers.operators.abstract import Operator
import torch

class LinearOperator(Operator):
    """
    Base for linear operators providing transpose and pseudo-inverse.
    """
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map: y = H(x)."""
        pass

    @abstractmethod
    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint: x = H^T(y)."""
        pass

    @abstractmethod
    def apply_pinv(self, y: torch.Tensor) -> torch.Tensor:
        """Pseudo-inverse: x = H^+(y)."""
        pass


class SVDLinearOperator(LinearOperator):
    """
    Generic linear operator using SVD: H = U @ diag(S) @ Vh.
    """
    def __init__(self, H: torch.Tensor, tol: float = 1e-6):
        u, s, vh = torch.linalg.svd(H, full_matrices=False)
        s = torch.where(s > tol, s, torch.zeros_like(s))
        self._u  = u      # [out_dim, k]
        self._s  = s      # [k]
        self._vh = vh     # [k, in_dim]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        proj = (x @ self._vh.T) * self._s
        return proj @ self._u.T

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        proj = (y @ self._u) * self._s
        return proj @ self._vh

    def apply_pinv(self, y: torch.Tensor) -> torch.Tensor:
        inv_s = torch.where(self._s > 0, 1.0/self._s, torch.zeros_like(self._s))
        proj  = (y @ self._u) * inv_s
        return proj @ self._vh