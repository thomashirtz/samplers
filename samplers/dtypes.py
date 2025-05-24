from typing import Sequence, TypeAlias

import torch
from torch import Tensor  # noqa

# Per-sample tensor shape (channels first, videos, 1-D signals …).
Shape: TypeAlias = Sequence[int] | torch.Size

# Anything accepted by `.to(device=...)` or `tensor.to(device=...)`.
Device: TypeAlias = torch.device | str | None

# A concrete torch dtype OR `None` (leave dtype unchanged).
DType: TypeAlias = torch.dtype | None

# Plain Python numeric literals – handy for parameters such as sigma, rate, etc.
Scalars: TypeAlias = int | float

# A Tensor or a scalar; useful for functions that accept either.
TensorLike: TypeAlias = Tensor | Scalars

# Optional RNG handle (pass `None` to use the global default RNG).
RNG: TypeAlias = torch.Generator | None
