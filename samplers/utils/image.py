import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image as _to_pil_image
from torchvision.transforms.functional import to_tensor as _to_tensor

from samplers.dtypes import Device, DType, Tensor


def tensor_to_pil(img_tensor: Tensor) -> Image.Image:
    """Convert a tensor in the range ``[-1, 1]`` to a PIL RGB image.

    Parameters
    ----------
    img_tensor : Tensor
        Tensor of shape ``(..., C, H, W)`` or ``(C, H, W)`` with values in
        ``[-1, 1]``.

    Returns
    -------
    PIL.Image.Image
        The resulting 8‑bit RGB image suitable for visualisation or saving.
    """
    # 1. Ensure the values are within the valid range.
    img_tensor = img_tensor.clamp(-1.0, 1.0)

    # 2. Rescale from ``[-1, 1]`` to ``[0, 1]``.
    img_tensor = (img_tensor + 1.0) * 0.5

    # 3. Move to CPU and cast to float32 for ``_to_pil_image``.
    img_tensor = img_tensor.to(dtype=torch.float32, device="cpu").contiguous()

    return _to_pil_image(img_tensor)


def pil_to_tensor(
    image: Image.Image,
    device: Device = None,
    dtype: DType = torch.float32,
) -> Tensor:
    """Convert a PIL image to a ``(C, H, W)`` tensor in the range ``[-1, 1]``.

    Parameters
    ----------
    image : PIL.Image.Image
        Input RGB image.
    device : torch.device or None, optional
        Target device; defaults to the current default device.
    dtype : torch.dtype, optional
        Desired floating‑point precision (default: ``torch.float32``).

    Returns
    -------
    Tensor
        Tensor representation of the image on the requested device/dtype with
        values in ``[-1, 1]``.
    """
    # 1. PIL → float tensor in ``[0, 1]`` with shape ``(C, H, W)``.
    tensor = _to_tensor(image)

    # 2. Rescale from ``[0, 1]`` to ``[-1, 1]``.
    tensor = (tensor * 2.0) - 1.0

    # 3. Transfer to the desired device and dtype.
    return tensor.to(device=device or torch.device("cpu"), dtype=dtype)
