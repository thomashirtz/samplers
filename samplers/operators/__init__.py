from .base import NonlinearOperator, Operator
from .identity import IdentityOperator
from .inpainting import (
    CenterInpaintingOperator,
    CenterOutpaintingOperator,
    InpaintingOperator,
    SidePaintingOperator,
    get_mask_inpaint_center,
    get_mask_side_painting,
)
from .linear import GeneralSVDOperator, LinearOperator, SVDOperator

__all__ = [
    "Operator",
    "NonlinearOperator",
    "IdentityOperator",
    "LinearOperator",
    "GeneralSVDOperator",
    "SVDOperator",
    "InpaintingOperator",
    "CenterInpaintingOperator",
    "CenterOutpaintingOperator",
    "SidePaintingOperator",
]
