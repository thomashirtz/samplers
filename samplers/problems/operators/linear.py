from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from samplers.operators import Operator
from samplers.utils import pad_zeros


class LinearOperator(Operator):
    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def apply_transpose(self, y: Tensor) -> Tensor:
        pass

    @abstractmethod
    def apply_pseudo_inverse(self, y: Tensor) -> Tensor:
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
        apply(x) – Hx
        apply_transpose(y) – Hᵀy
        apply_pseudo_inverse(y) – H⁺y
    """

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns ``(m, n)`` – *output* and *input* dimensions."""
        raise NotImplementedError

    # ---- Low‑level factor products ------------------------------------
    def apply_U(self, x: Tensor) -> Tensor:  # noqa: D401
        """Computes ``U · x`` where ``U`` has shape ``(m, k)`` and ``x`` is
        ``(..., k)``."""
        raise NotImplementedError

    def apply_U_transpose(self, x: Tensor) -> Tensor:  # noqa: D401
        """Computes ``Uᵀ · x``  with ``x`` of shape ``(..., m)`` → ``(...,
        k)``."""
        raise NotImplementedError

    def apply_V(self, x: Tensor) -> Tensor:  # noqa: D401
        """Computes ``V · x`` where ``V`` is ``(n, k)`` and ``x`` is ``(...,
        k)`` → ``(..., n)``."""
        raise NotImplementedError

    def apply_V_transpose(self, x: Tensor) -> Tensor:  # noqa: D401
        """Computes ``Vᵀ · x`` with ``x`` of shape ``(..., n)`` → ``(...,
        k)``."""
        raise NotImplementedError

    def get_singular_values(self) -> Tensor:  # noqa: D401
        """Returns the 1‑D spectrum ``s`` (shape ``(k,)``)."""
        raise NotImplementedError

    def apply(self, x: Tensor) -> Tensor:  # noqa: D401
        """Computes *Hx* via factor products (no need for sub‑class
        override)."""
        z = self.apply_V_transpose(x)  # (..., k)
        s = self.get_singular_values()  # (k,)
        z = z * s  # (..., k)
        return self.apply_U(z)  # (..., m)

    def apply_transpose(self, y: Tensor) -> Tensor:  # noqa: D401
        """Computes *Hᵀy* via factor products."""
        z = self.apply_U_transpose(y)  # (..., k)
        s = self.get_singular_values()
        z = z * s  # (..., k)
        n = self.shape[1]
        return self.apply_V(pad_zeros(z, n))  # (..., n)

    def apply_pseudo_inverse(self, y: Tensor) -> Tensor:  # noqa: D401
        """Computes the Moore–Penrose pseudo‑inverse product *H⁺y*."""
        z = self.apply_U_transpose(y)  # (..., k)
        s = self.get_singular_values()
        inv_s = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
        z = z * inv_s  # (..., k)
        n = self.shape[1]
        return self.apply_V(pad_zeros(z, n))  # (..., n)


class GeneralSVDOperator(SVDOperator):
    def __init__(self, U, s, Vh):
        super().__init__()
        self._m, self._n = U.shape[0], Vh.shape[1]
        self.register_buffer("_U", U)
        self.register_buffer("_singular_values", s)
        self.register_buffer("_V_transpose", Vh)

    def apply_U(self, x):
        return x @ self._U.t()

    def apply_U_transpose(self, x):
        return x @ self._U

    def apply_V(self, x: Tensor) -> Tensor:  # x : (..., k)
        return x @ self._V_transpose  # (..., n)

    def apply_V_transpose(self, x: Tensor) -> Tensor:  # x : (..., n)
        return x @ self._V_transpose.t()  # (..., k)

    def get_singular_values(self):
        return self._singular_values

    @property
    def shape(self) -> Tuple[int, int]:
        """Output dimension m, input dimension n."""
        return self._m, self._n
