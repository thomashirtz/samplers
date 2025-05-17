from abc import abstractmethod
from samplers.operators.abstract import Operator
import torch
import torch
from torch import Tensor
from typing import Tuple

class LinearOperator(Operator):
    """
    Base for linear operators providing transpose and pseudo-inverse.
    """
    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Forward map: y = H(x)."""
        pass

    @abstractmethod
    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint: x = H^T(y)."""
        pass

    @abstractmethod
    def apply_pinv(self, y: Tensor) -> Tensor:
        """Pseudo-inverse: x = H^+(y)."""
        pass


class SVDLinearOperator(LinearOperator):
    """
    Linear operator given by its (thin) SVD  H = U @ diag(s) @ Vh.

    Shapes (thin SVD, rank = k)                       Example
    -------------------------------------------------  ---------------------
      U   : (m, k)    (left singular vectors)          (1000, 128)
      s   : (k,)      (non-negative singular values)   (128,)
      Vh  : (k, n)    (right singular vectors †)       (128, 256)

    †  `Vh` is *V transposed* (as returned by ``torch.linalg.svd``).

    Notes
    -----
    • All three pieces are registered as *buffers*, not *parameters* —
      they follow the module onto new devices/dtypes but are ignored
      by optimizers and `.state_dict(load)`.

    • The operator implements:
        forward(x)        – Hx
        apply(x)          – Hx   (alias of forward)
        apply_transpose(y)– Hᵀy
        apply_pinv(y)     – H⁺y  (Moore–Penrose pseudoinverse)

    • Multiplication is also available with the `@` operator:
        z = H @ x         # same as H(x)
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
        """Hx for batched or un-batched `x` (…, n)."""
        proj = (x @ self._vh.T) * self._s
        return proj @ self._u.T

    forward = apply

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Hᵀy for batched or un-batched `y` (…, m)."""
        proj = (y @ self._u) * self._s
        return proj @ self._vh

    def apply_pinv(self, y: Tensor) -> Tensor:
        """H⁺y – pseudo-inverse action (handles rank deficiency)."""
        inv_s = torch.where(self._s > 0, 1.0 / self._s, torch.zeros_like(self._s))
        proj = (y @ self._u) * inv_s
        return proj @ self._vh

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.apply(other)

    @classmethod
    def from_matrix(cls, H: Tensor, *, full_matrices: bool = False) -> "SVDLinearOperator":
        """
        Factory that computes the thin SVD with `torch.linalg.svd`
        and returns a ready-to-use operator.
        """
        u, s, vh = torch.linalg.svd(H, full_matrices=full_matrices)
        return cls(u, s, vh)

    @staticmethod
    def _check_shapes(u: Tensor, s: Tensor, vh: Tensor) -> None:
        if u.dim() != 2 or vh.dim() != 2 or s.dim() != 1:
            raise ValueError("u and vh must be 2-D, s must be 1-D")
        m, k1 = u.shape
        k2, n = vh.shape
        if k1 != k2 or k1 != s.numel():
            raise ValueError(
                f"Shape mismatch: U is {u.shape}, S is {s.shape}, Vh is {vh.shape}"
            )
