import torch

from samplers.dtypes import Device, Shape, Tensor

from .linear import SVDOperator


class InpaintingOperator(SVDOperator):
    """Inpainting/outpainting operator.

    This operator handles keeping or masking pixels in an image. All
    methods are batch-agnostic and support any number of leading
    dimensions.
    """

    def __init__(
        self,
        x_shape: Shape,
        mask: Tensor,
        flatten: bool = True,
        device: Device = None,
    ):
        """Initialize the inpainting operator.

        Args:
            x_shape: Shape of the input image (e.g., (C, H, W)).
            mask: Boolean mask where True indicates missing/masked pixels.
            flatten: If True, forward returns only kept pixels as a vector.
                     If False, forward returns full image with zeros at masked positions.
            device: Device on which to place the operator.
        """

        # User flags / shapes
        self._x_shape_internal: Shape = tuple(x_shape)
        self.flatten: bool = bool(flatten)

        # Ensure boolean mask, broadcast if (H,W) for (C,H,W)
        target_device = torch.device(device) if device is not None else mask.device
        mask = mask.to(target_device)
        if mask.dtype != torch.bool:
            mask = mask.ne(0)

        if mask.shape != self._x_shape_internal:
            raise ValueError(
                f"Mask shape incompatible with x_shape: {mask.shape} vs. {self._x_shape_internal}."
            )

        # Kept indices & singular values
        flat_mask = mask.flatten()  # (n,)
        kept_indices_tmp = torch.nonzero(~flat_mask, as_tuple=False).squeeze(1)  # (m,)
        singular_values_tmp = torch.ones(
            kept_indices_tmp.numel(), dtype=torch.float32, device=target_device
        )

        self._m_dim: int = kept_indices_tmp.numel()
        self._n_dim: int = flat_mask.numel()

        # Hand these to the parent before it infers shapes
        self._kept_indices = kept_indices_tmp
        self._singular_values = singular_values_tmp
        super().__init__(x_shape=self._x_shape_internal, device=target_device)

        # Now turn them into proper buffers
        del self._kept_indices, self._singular_values
        self.register_buffer("mask", mask)
        self.register_buffer("_kept_indices", kept_indices_tmp)
        self.register_buffer("_singular_values", singular_values_tmp)

    @staticmethod
    def _expand_kept_indices(batch_dims: tuple[int, ...], idx_1d: Tensor) -> Tensor:
        """Expand indices to match batch dimensions.

        Args:
            batch_dims: Tuple of batch dimensions.
            idx_1d: 1D tensor of indices.

        Returns:
            Expanded indices tensor.
        """
        view_shape = (1,) * len(batch_dims) + (idx_1d.numel(),)
        return idx_1d.view(view_shape).expand(*batch_dims, -1)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the operator as (m, n).

        Returns:
            Tuple of (m, n) where m is the number of kept pixels and n is the total number of pixels.
        """
        return self._m_dim, self._n_dim

    def apply(self, x: Tensor) -> Tensor:
        """Forward map: $y = A(x)$

        Args:
            x: Input tensor with shape (*batch, *x_shape).

        Returns:
            If flatten=True: Kept-pixel vector (parent implementation).
            If flatten=False: Full image with missing pixels zeroed.
        """
        if self.flatten:
            return super().apply(x)  # parent uses $V^T Σ U^T$ = gather kept pixels

        # Full image: just zero out the masked locations
        batch_dims = x.shape[: -len(self._x_shape_internal)]
        expanded_mask = self.mask.view((1,) * len(batch_dims) + self.mask.shape)
        expanded_mask = expanded_mask.expand(*batch_dims, *self.mask.shape)
        return x.masked_fill(expanded_mask, 0)

    def apply_transpose(self, y: Tensor) -> Tensor:
        """Adjoint/transpose map: $x = A^T(y)$

        Args:
            y: Input tensor with shape (*batch, *y_shape).

        Returns:
            If flatten=True: Parent implementation (scatter kept → full image).
            If flatten=False: Zero out masked pixels (ensures correct gradients).
        """
        if self.flatten:
            return super().apply_transpose(y)

        batch_dims = y.shape[: -len(self._x_shape_internal)]
        expanded_mask = self.mask.view((1,) * len(batch_dims) + self.mask.shape)
        expanded_mask = expanded_mask.expand(*batch_dims, *self.mask.shape)
        return y.masked_fill(expanded_mask, 0)

    # Make pseudo-inverse equal to the transpose (Moore-Penrose pinv = adjoint here)
    apply_pseudo_inverse = apply_transpose

    def apply_V_transpose(self, x: Tensor) -> Tensor:
        """Apply $V^T$ component of SVD to input x.

        Args:
            x: Input tensor with shape (*batch, *x_shape).

        Returns:
            Output tensor with shape (*batch, m).
        """
        sample_rank = len(self._x_shape_internal)
        batch_dims = x.shape[:-sample_rank]
        x_flat = x.reshape(*batch_dims, -1)
        idx = self._expand_kept_indices(batch_dims, self._kept_indices)
        return x_flat.gather(dim=-1, index=idx)  # (*batch, m)

    def apply_U(self, z: Tensor) -> Tensor:
        """Apply $U$ component of SVD to input z.

        Args:
            z: Input tensor with shape (*batch, m).

        Returns:
            Output tensor with shape (*batch, m).
        """
        return z

    def apply_U_transpose(self, y: Tensor) -> Tensor:
        """Apply $U^T$ component of SVD to input y.

        Args:
            y: Input tensor with shape (*batch, m).

        Returns:
            Output tensor with shape (*batch, m).
        """
        return y

    def apply_V(self, z_kept: Tensor) -> Tensor:
        """Apply $V$ component of SVD to input z_kept.

        Args:
            z_kept: Input tensor with shape (*batch, m).

        Returns:
            Output tensor with shape (*batch, *x_shape).
        """
        batch_dims = z_kept.shape[:-1]
        flat = torch.zeros(
            *batch_dims,
            self._n_dim,
            device=z_kept.device,
            dtype=z_kept.dtype,
        )
        idx = self._expand_kept_indices(batch_dims, self._kept_indices)
        flat.scatter_(dim=-1, index=idx, src=z_kept)
        return flat.reshape(*batch_dims, *self._x_shape_internal)

    def get_singular_values(self) -> Tensor:
        """Return the singular values of the operator.

        Returns:
            Tensor of singular values.
        """
        return self._singular_values


class CenterInpaintingOperator(InpaintingOperator):
    """Inpaints the center region of an image, masking out a central
    rectangle."""

    def __init__(self, x_shape: Shape, paint_fraction: float = 0.5, device: Device = None):
        """Create a center inpainting operator.

        Args:
            x_shape: Shape of the input image (e.g., (C, H, W)).
            paint_fraction: Fraction of each dimension to mask (inpaint) in the center.
                            0 → nothing, 1 → whole image.
            device: Device on which to place the mask.
        """
        if not (0.0 <= paint_fraction <= 1.0):
            raise ValueError("paint_fraction must be in [0, 1]")
        # Compute start/end as centered span
        start = (1.0 - paint_fraction) / 2.0
        end = (1.0 + paint_fraction) / 2.0

        mask = get_mask_inpaint_center(x_shape, start, end, device=device)
        super().__init__(x_shape, mask)


class CenterOutpaintingOperator(InpaintingOperator):
    """Outpaints the periphery of an image, keeping a centered rectangle."""

    def __init__(self, x_shape: Shape, keep_fraction: float = 0.5, device: Device = None):
        """Create a center outpainting operator.

        Args:
            x_shape: Shape of the input image (e.g., (C, H, W)).
            keep_fraction: Fraction of each dimension to keep in the center.
                           0 → nothing, 1 → whole image.
            device: Device on which to place the mask.
        """
        if not (0.0 <= keep_fraction <= 1.0):
            raise ValueError("keep_fraction must be in [0, 1]")
        # Compute start/end for the kept center
        start = (1.0 - keep_fraction) / 2.0
        end = (1.0 + keep_fraction) / 2.0

        center_mask = get_mask_inpaint_center(x_shape, start, end, device=device)
        outpaint_mask = ~center_mask
        super().__init__(x_shape, outpaint_mask)


class SidePaintingOperator(InpaintingOperator):
    """Inpaints one side of an image, masking out a vertical slice."""

    def __init__(
        self,
        x_shape: Shape,
        paint_fraction: float = 0.5,
        left: bool = True,
        device: Device = None,
    ):
        """Create a side painting operator.

        Args:
            x_shape: Shape of the input image (e.g., (C, H, W)).
            paint_fraction: Fraction of the width to mask on the chosen side.
                            0 → nothing, 1 → full width.
            left: If True, mask the left side; otherwise, mask the right side.
            device: Device on which to place the mask.
        """
        if not (0.0 <= paint_fraction <= 1.0):
            raise ValueError("paint_fraction must be in [0, 1]")
        mask = get_mask_side_painting(x_shape, paint_fraction, left, device=device)
        super().__init__(x_shape, mask)


def get_mask_inpaint_center(
    image_shape: Shape,
    start_pct: float = 0.25,
    end_pct: float = 0.75,
    device: Device = None,
) -> Tensor:
    """Generate a boolean mask for center inpainting.

    Args:
        image_shape: Tuple like (..., H, W).
        start_pct: Starting percentage for inpainting region.
        end_pct: Ending percentage for inpainting region.
        device: Device for the generated mask.

    Returns:
        Boolean mask where True indicates pixels to be inpainted (missing regions)
        and False indicates kept regions.
    """
    if not (0 <= start_pct < end_pct <= 1):
        raise ValueError("start_pct and end_pct must satisfy 0 <= start_pct < end_pct <= 1")

    h, w = image_shape[-2], image_shape[-1]
    start_h_px, end_h_px = int(h * start_pct), int(h * end_pct)
    start_w_px, end_w_px = int(w * start_pct), int(w * end_pct)

    # Mask is False for kept regions by default
    mask = torch.zeros(image_shape, dtype=torch.bool, device=device)
    mask[..., start_h_px:end_h_px, start_w_px:end_w_px] = True
    return mask


def get_mask_side_painting(
    image_shape: Shape,
    pct: float = 0.50,
    left: bool = True,
    device: Device = None,
) -> Tensor:
    """Generate a boolean mask for side painting.

    Args:
        image_shape: Tuple like (..., H, W).
        pct: Fraction of width to mask.
        left: Whether to mask the left or right side.
        device: Device for the generated mask.

    Returns:
        Boolean mask where True indicates pixels to be inpainted (missing regions)
        and False indicates kept regions.
    """
    if not (0 < pct <= 1):
        raise ValueError("pct must satisfy 0 < pct <= 1")

    h, w = image_shape[-2], image_shape[-1]
    mask_width = int(w * pct)

    mask = torch.zeros(image_shape, dtype=torch.bool, device=device)  # False for kept

    if left:
        mask[..., :mask_width] = True  # True for missing
    else:
        mask[..., -mask_width:] = True
    return mask
