import torch

from samplers.dtypes import Device, Shape, Tensor

from .linear import SVDOperator


class InpaintingOperator(SVDOperator):
    """Inpainting operator implemented using SVDOperator.

    This operator selects the "kept" pixels from an image.
    - Mask: True for missing pixels, False for kept pixels.
    - apply(x): Extracts kept pixels from x and returns them as a flat vector.
    - apply_transpose(y_kept): Embeds kept pixel values y_kept into a full image.
    - apply_pseudo_inverse(y_kept): Same as apply_transpose for this operator.
    """

    def __init__(self, x_shape: Shape, mask: Tensor, device: Device = None):
        """
        x_shape: Shape of the input image (e.g., (C, H, W)).
        mask: Tensor representing the mask. True for missing pixels, False for kept pixels.
              If not boolean, it will be converted (non-zero to True, zero to False).
              Shape must be compatible with x_shape (either same shape, or (H,W) for (C,H,W) x_shape).
        """
        self._x_shape_internal = tuple(x_shape)
        target_device = torch.device(device) if device is not None else mask.device

        # Convert mask to boolean if it isn't already.
        # Assumes for numerical masks: 0 is False (kept), non-zero is True (missing).
        if mask.dtype != torch.bool:
            if mask.is_floating_point() or mask.dtype in [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ]:
                print(
                    f"Warning: Converting input mask of dtype {mask.dtype} to torch.bool. "
                    "Assuming 0 represents kept pixels (False) and non-zero represents missing pixels (True)."
                )
                mask = mask.ne(0)  # Equivalent to mask != 0, results in a boolean tensor
            else:
                raise ValueError(
                    f"Mask dtype {mask.dtype} is not torch.bool and not a recognized numerical type for conversion."
                )
        mask = mask.to(target_device)

        processed_mask = self._prepare_mask(mask, self._x_shape_internal)

        current_device = processed_mask.device
        flat_mask = processed_mask.flatten()  # Boolean tensor

        # Kept indices are where mask is False
        temp_kept_indices = torch.nonzero(~flat_mask).squeeze(1).to(current_device)
        # Singular values for kept pixels are 1.0. Default to float32.
        # Dtype of s_values can affect output dtype of apply() if input x is integer.
        temp_s_values = torch.ones(
            temp_kept_indices.shape[0], dtype=torch.float32, device=current_device
        )

        self._m_dim = temp_kept_indices.shape[0]  # Output dimension (number of kept pixels)
        self._n_dim = flat_mask.shape[0]  # Input dimension (total pixels in x)

        # Call super().__init__ AFTER essential attributes for SVD methods are set (as temp_*)
        # Operator.__init__ will call self.apply() via _infer_y_shape.
        # The super actually need _kept_indices and _s_singular_values to infer y_shape, that is
        #  why they are created before doing the super then deleted.
        self._kept_indices = temp_kept_indices
        self._s_singular_values = temp_s_values
        super().__init__(self._x_shape_internal)
        del self._kept_indices
        del self._s_singular_values

        # Register buffers for proper nn.Module behavior (device moving, state_dict, etc.)
        self.register_buffer("mask", processed_mask)  # Boolean mask
        self.register_buffer("_kept_indices", temp_kept_indices)
        self.register_buffer("_s_singular_values", temp_s_values)  # Float32 singular values

    @staticmethod
    def _prepare_mask(mask_in: Tensor, target_x_shape: Shape) -> Tensor:
        """Validates and prepares the boolean mask to match target_x_shape."""
        if mask_in.shape == target_x_shape:
            return mask_in

        # Handle common case: (H,W) mask for (C,H,W) x_shape
        if len(target_x_shape) == 3 and mask_in.ndim == 2:
            c, h, w = target_x_shape
            if mask_in.shape == (h, w):
                return mask_in.unsqueeze(0).expand(c, h, w)

        raise ValueError(
            f"Mask shape {mask_in.shape} is not compatible with x_shape {target_x_shape}. "
            "Mask must either be same shape as x_shape, or (H,W) for x_shape (C,H,W)."
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Returns (m, n) - output (kept_pixels_count) and input (total_pixels_count) dimensions."""
        return self._m_dim, self._n_dim

    def get_singular_values(self) -> Tensor:
        """Returns the singular values (all 1.0s for kept pixels)."""
        return self._s_singular_values

    def apply_V_transpose(self, x: Tensor) -> Tensor:
        """Extracts kept pixels from x.

        (Vh part of H = U S Vh)
        x: Input image tensor (Batch, *x_shape). Its dtype is preserved.
        Returns: Tensor of kept pixels (Batch, num_kept_pixels).
        """
        batch_size = x.shape[0]
        # Flatten x while preserving batch dimension. Resulting dtype is same as x.
        x_flat = x.reshape(batch_size, -1)

        # kept_indices is (num_kept_pixels,). Expand for batch gathering.
        indices_for_gather = self._kept_indices.unsqueeze(0).expand(batch_size, -1)
        # gather preserves dtype of x_flat.
        return x_flat.gather(1, indices_for_gather)

    def apply_U(self, z: Tensor) -> Tensor:
        """Applies U matrix.

        For this operator, U is effectively identity.
        z: Tensor (Batch, num_kept_pixels). Dtype is preserved.
        Returns: Tensor (Batch, num_kept_pixels).
        """
        return z

    def apply_U_transpose(self, y: Tensor) -> Tensor:
        """Applies U_transpose.

        For this operator, U_transpose is effectively identity.
        y: Tensor (Batch, num_kept_pixels). Dtype is preserved.
        Returns: Tensor (Batch, num_kept_pixels).
        """
        return y

    def apply_V(self, z_kept: Tensor) -> Tensor:
        """Embeds kept pixel values back into a full image.

        (V part of H = U S Vh)
        z_kept: Tensor of kept pixel values (Batch, num_kept_pixels).
        Returns: Full image tensor with kept values placed, zeros elsewhere (Batch, *self.x_shape).
                 Output dtype matches z_kept.
        """
        batch_size = z_kept.shape[0]

        # Create a zero tensor with the target full shape, matching z_kept's dtype and device.
        reconstructed_flat = torch.zeros(
            batch_size, self._n_dim, device=z_kept.device, dtype=z_kept.dtype
        )

        indices_for_scatter = self._kept_indices.unsqueeze(0).expand(batch_size, -1)
        reconstructed_flat.scatter_(1, indices_for_scatter, z_kept)

        # Reshape to original image dimensions (per batch item).
        # self.x_shape is set by Operator base class.
        return reconstructed_flat.reshape(batch_size, *self.x_shape)


class CenterInpaintingOperator(InpaintingOperator):
    """Inpaints the center region of an image, masking out a central
    rectangle."""

    def __init__(self, x_shape: Shape, paint_fraction: float = 0.5, device: Device = None):
        """
        x_shape:         Shape of the input image (e.g., (C, H, W)).
        paint_fraction: Fraction of each dimension to mask (inpaint) in the center. 0 → nothing, 1 → whole image.
        device:          Device on which to place the mask.
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
        """
        x_shape:        Shape of the input image (e.g., (C, H, W)).
        keep_fraction:  Fraction of each dimension to keep in the center. 0 → nothing, 1 → whole image.
        device:         Device on which to place the mask.
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
        self, x_shape: Shape, paint_fraction: float = 0.5, left: bool = True, device: Device = None
    ):
        """
        x_shape:         Shape of the input image (e.g., (C, H, W)).
        paint_fraction: Fraction of the width to mask on the chosen side. 0 → nothing, 1 → full width.
        left:            If True, mask the left side; otherwise, mask the right side.
        device:          Device on which to place the mask.
        """
        if not (0.0 <= paint_fraction <= 1.0):
            raise ValueError("paint_fraction must be in [0, 1]")
        mask = get_mask_side_painting(x_shape, paint_fraction, left, device=device)
        super().__init__(x_shape, mask)


def get_mask_inpaint_center(
    image_shape: Shape,
    start_pct: float = 0.25,
    end_pct: float = 0.75,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generates a boolean mask for center inpainting.

    Mask is True for missing/inpainted region, False for kept region.
    image_shape: Tuple like (C, H, W) or (H, W).
    """
    if not (0 <= start_pct < end_pct <= 1):
        raise ValueError("start_pct and end_pct must satisfy 0 <= start_pct < end_pct <= 1")

    h, w = image_shape[-2], image_shape[-1]
    start_h_px, end_h_px = int(h * start_pct), int(h * end_pct)
    start_w_px, end_w_px = int(w * start_pct), int(w * end_pct)

    # Mask is False for kept regions by default
    mask = torch.zeros(image_shape, dtype=torch.bool, device=device)

    if len(image_shape) == 2:  # H, W
        mask[start_h_px:end_h_px, start_w_px:end_w_px] = True  # True for missing
    elif len(image_shape) == 3:  # C, H, W
        mask[:, start_h_px:end_h_px, start_w_px:end_w_px] = True
    else:
        raise ValueError(f"Unsupported image_shape dimension: {len(image_shape)}. Expected 2 or 3.")
    return mask


def get_mask_side_painting(
    image_shape: Shape, pct: float = 0.50, left: bool = True, device: Device = None
) -> Tensor:
    """Generates a boolean mask for side painting.

    Mask is True for missing/inpainted region, False for kept region.
    image_shape: Tuple like (C, H, W) or (H, W).
    """
    if not (0 < pct <= 1):
        raise ValueError("pct must satisfy 0 < pct <= 1")

    h, w = image_shape[-2], image_shape[-1]
    mask_width = int(w * pct)

    mask = torch.zeros(image_shape, dtype=torch.bool, device=device)  # False for kept

    if len(image_shape) == 2:  # H, W
        if left:
            mask[:, :mask_width] = True  # True for missing
        else:
            mask[:, -mask_width:] = True
    elif len(image_shape) == 3:  # C, H, W
        if left:
            mask[..., :, :mask_width] = True
        else:
            mask[..., :, -mask_width:] = True
    else:
        raise ValueError(f"Unsupported image_shape dimension: {len(image_shape)}. Expected 2 or 3.")
    return mask
