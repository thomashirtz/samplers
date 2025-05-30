from abc import abstractmethod
from typing import Tuple

import torch

from samplers.dtypes import Device, Shape, Tensor

from .base import Operator


class LinearOperator(Operator):
    """Linear forward operator with optional adjoint and pseudo-inverse.

    This base class implements linear operators where $A$ is a matrix (or linear map).
    For efficiency, the matrix may never be explicitly formed.

    Required:
      apply(x) - forward map $y = Ax$

    Optional:
      apply_transpose(y) - adjoint/transpose $x = A^T y$
      apply_pseudo_inverse(y) - pseudo-inverse $x = A^+ y$ (if well-defined)
    """

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint/transpose map: $x = A^T(y)$

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            Output tensor with shape (*batch, *x_shape).
        """
        return super().apply_transpose(y)  # May raise NotImplementedError

    def apply_pseudo_inverse(self, y: Tensor) -> Tensor:
        """Pseudo-inverse map: $x = A^+(y)$

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            Output tensor with shape (*batch, *x_shape).
        """
        return super().apply_pseudo_inverse(y)  # May raise NotImplementedError


class SVDOperator(LinearOperator):
    """Linear operator defined via SVD: $H = U \Sigma V^T$.

    Shapes (thin SVD, rank = k):
      $U$ : $(m, k)$ – left singular vectors
      $\Sigma$ : $(k,)$ – singular values (non-negative)
      $V^T$ : $(k, n)$ – right singular vectors (transposed)

    Implements:
      apply(x) – $Hx$
      apply_transpose(y) – $H^T y$
      apply_pseudo_inverse(y) – $H^+ y$
    """

    def __init__(self, x_shape: Shape, device: Device = None) -> None:
        """Initialize the SVD operator.

        Args:
            x_shape: Shape of the input tensor.
            device: Device on which to place the operator.
        """
        super().__init__(x_shape=x_shape, device=device)

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns $(m, n)$ – output and input dimensions.

        Returns:
            Tuple of (m, n) where m is the output dimension and n is the input dimension.
        """
        raise NotImplementedError("Subclasses must implement shape property")

    @abstractmethod
    def apply_U(self, z: Tensor) -> Tensor:
        """Computes $U \cdot z$ where $U$ has shape $(m, k)$ and $z$ is
        $(*batch, k)$.

        Args:
            z: Input tensor with shape (*batch, k).

        Returns:
            Output tensor with shape (*batch, m).
        """
        pass

    @abstractmethod
    def apply_U_transpose(self, y: Tensor) -> Tensor:
        """Computes $U^T \cdot y$ with $y$ of shape $(*batch, m) \to (*batch,
        k)$.

        Args:
            y: Input tensor with shape (*batch, m).

        Returns:
            Output tensor with shape (*batch, k).
        """
        pass

    @abstractmethod
    def apply_V(self, z: Tensor) -> Tensor:
        """Computes $V \cdot z$ where $V$ is $(n, k)$ and $z$ is $(*batch, k)
        \to (*batch, n)$.

        Args:
            z: Input tensor with shape (*batch, k).

        Returns:
            Output tensor with shape (*batch, n).
        """
        pass

    @abstractmethod
    def apply_V_transpose(self, x: Tensor) -> Tensor:
        """Computes $V^T \cdot x$ with $x$ of shape $(*batch, n) \to (*batch,
        k)$.

        Args:
            x: Input tensor with shape (*batch, n).

        Returns:
            Output tensor with shape (*batch, k).
        """
        pass

    @abstractmethod
    def get_singular_values(self) -> Tensor:
        """Returns the 1-D spectrum $s$ (shape $(k,)$).

        Returns:
            Tensor of singular values with shape (k,).
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Computes $Hx$ via factor products.

        Args:
            x: Input tensor with shape (*batch, *x_shape).

        Returns:
            Output tensor with shape (*batch, *y_shape).
        """
        z = self.apply_V_transpose(x)  # $(*batch, k)$
        z = z * self.get_singular_values()  # $(*batch, k)$
        return self.apply_U(z)  # $(*batch, m)$

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Computes $H^T y$ via factor products.

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            Output tensor with shape (*batch, *x_shape).
        """
        z = self.apply_U_transpose(y)  # $(*batch, k)$
        z = z * self.get_singular_values()  # $(*batch, k)$
        return self.apply_V(z)  # $(*batch, n)$

    def apply_pseudo_inverse(self, y: Tensor) -> Tensor:
        """Computes the Moore–Penrose pseudo-inverse product $H^+ y$.

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            Output tensor with shape (*batch, *x_shape).
        """
        z = self.apply_U_transpose(y)  # $(*batch, k)$
        s = self.get_singular_values()
        # Invert non-zero singular values, leave zeros as zeros
        s_inv = torch.zeros_like(s)
        mask = s > 0
        s_inv[mask] = 1.0 / s[mask]
        z = z * s_inv  # $(*batch, k)$
        return self.apply_V(z)  # $(*batch, n)$


class GeneralSVDOperator(SVDOperator):
    """SVD operator with explicit matrices $U$, $\Sigma$, and $V^T$.

    This class allows creation of an operator from an explicit SVD
    decomposition, with $U$, $\Sigma$, and $V^T$ provided as tensors.
    """

    def __init__(self, U: Tensor, s: Tensor, Vh: Tensor):
        """Initialize from explicit SVD matrices.

        Args:
            U: Left singular vectors, shape (m, k)
            s: Singular values, shape (k,)
            Vh: Right singular vectors transposed, shape (k, n)
        """
        super().__init__(x_shape=(Vh.shape[1],))
        self._m, self._n = U.shape[0], Vh.shape[1]
        self.register_buffer("_U", U)
        self.register_buffer("_singular_values", s)
        self.register_buffer("_V_transpose", Vh)

    def apply_U(self, x: Tensor) -> Tensor:
        """Apply $U$ component of SVD to input x.

        Args:
            x: Input tensor with shape (*batch, k).

        Returns:
            Output tensor with shape (*batch, m).
        """
        return x @ self._U.t()

    def apply_U_transpose(self, x: Tensor) -> Tensor:
        """Apply $U^T$ component of SVD to input x.

        Args:
            x: Input tensor with shape (*batch, m).

        Returns:
            Output tensor with shape (*batch, k).
        """
        return x @ self._U

    def apply_V(self, x: Tensor) -> Tensor:
        """Apply $V$ component of SVD to input x.

        Args:
            x: Input tensor with shape (*batch, k).

        Returns:
            Output tensor with shape (*batch, n).
        """
        return x @ self._V_transpose  # (*batch, n)

    def apply_V_transpose(self, x: Tensor) -> Tensor:
        """Apply $V^T$ component of SVD to input x.

        Args:
            x: Input tensor with shape (*batch, n).

        Returns:
            Output tensor with shape (*batch, k).
        """
        return x @ self._V_transpose.t()  # (*batch, k)

    def get_singular_values(self) -> Tensor:
        """Return the singular values of the operator.

        Returns:
            Tensor of singular values with shape (k,).
        """
        return self._singular_values

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the operator as (m, n).

        Returns:
            Tuple of (m, n) where m is the output dimension and n is the input dimension.
        """
        return self._m, self._n
