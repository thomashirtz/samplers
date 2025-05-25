import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image as _to_pil_image
from torchvision.transforms.functional import to_tensor as _to_tensor

from samplers.dtypes import Device, DType, Tensor


def tensor_to_pil(img_tensor: Tensor) -> Image:
    """Convert a torch tensor to PIL Image.

    Arguments:
    - img_tensor: tensor of shape [..., C, H, W] or [C, H, W], in float [0,1] or uint8.
    Returns:
    - PIL Image.
    """
    img_tensor = img_tensor.add(1.0).div(2.0)
    return _to_pil_image(img_tensor)


def pil_to_tensor(
    image: Image.Image,
    device: Device = None,
    dtype: DType = torch.float32,
) -> Tensor:
    """Convert a PIL image to a C×H×W tensor with values in [−1, 1]."""
    # 1) PIL → float tensor in [0,1], shape (C, H, W)
    tensor = _to_tensor(image)

    # 2) scale [0,1] → [−1,1]
    tensor = tensor.mul(2.0).sub(1.0)

    # 3) ensure desired device & dtype
    return tensor.to(device=device or tensor.device, dtype=dtype)
