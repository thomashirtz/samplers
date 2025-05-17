from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from samplers.operators.abstract import Operator


class LinearOperator(Operator):
    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def apply_transpose(self, y: Tensor) -> Tensor:
        pass

    @abstractmethod
    def apply_pinv(self, y: Tensor) -> Tensor:
        pass


class SVDOperator(LinearOperator):
    """
    Linear operator defined via SVD: H = U @ diag(s) @ Vh.

    Shapes (thin SVD, rank = k)
    ---------------------------
      U   : (m, k)    – left singular vectors
      s   : (k,)      – singular values (non-negative)
      Vh  : (k, n)    – right singular vectors (transposed)

    Notes
    -----
    • U, s, and Vh are registered as *buffers*, not *parameters*.
    • Implements:
        apply(x)           – Hx
        apply_transpose(y) – Hᵀy
        apply_pinv(y)      – H⁺y
        __matmul__         – H @ x (alias of apply)
    """

    def __init__(self, u: Tensor, s: Tensor, vh: Tensor) -> None:
        super().__init__()
        self._check_shapes(u, s, vh)

        self.register_buffer("_u", u)
        self.register_buffer("_s", s)
        self.register_buffer("_vh", vh)

        self._m, self._k = u.shape
        self._n = vh.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._m, self._n

    def apply(self, x: Tensor) -> Tensor:
        """
        Forward map:
        x: (..., n)
        Hx = U @ diag(s) @ Vh @ x
           = ((x @ Vh.T) * s) @ U.T
        """
        proj = (x @ self._vh.T) * self._s
        # fixme those implementation might change in the future, I want to keep it clean for now and
        #  see how I can integrate all the operator with a simple structure
        return proj @ self._u.T

    def apply_transpose(self, y: Tensor) -> Tensor:
        """
        Adjoint map:
        y: (..., m)
        Hᵀy = V @ diag(s) @ Uᵀ @ y
           = ((y @ self._u) * s) @ Vh
        """
        proj = (y @ self._u) * self._s
        return proj @ self._vh

    def apply_pinv(self, y: Tensor) -> Tensor:
        """
        Pseudo-inverse map:
        y: (..., m)
        H⁺y = V @ diag(1/s) @ Uᵀ @ y  (zeros handled via inv_s mask)
        """
        inv_s = torch.where(self._s > 0, 1.0 / self._s, torch.zeros_like(self._s))
        proj = (y @ self._u) * inv_s
        return proj @ self._vh

    __matmul__ = apply
    forward = apply

    @classmethod
    def from_matrix(cls, H: Tensor, full_matrices: bool = False) -> "SVDOperator":
        """
        Factory method: compute thin SVD using torch.linalg.svd and return operator.
        """
        u, s, vh = torch.linalg.svd(H, full_matrices=full_matrices)
        return cls(u, s, vh)

    @staticmethod
    def _check_shapes(u: Tensor, s: Tensor, vh: Tensor) -> None:
        if u.dim() != 2 or vh.dim() != 2 or s.dim() != 1:
            raise ValueError("u,vh must be 2-D, s must be 1-D")
        m, k1 = u.shape
        k2, n = vh.shape
        if k1 != k2 or k1 != s.numel():
            raise ValueError(f"Shape mismatch: U{u.shape}, S{s.shape}, Vh{vh.shape}")
