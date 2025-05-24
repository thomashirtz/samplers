from .base import NonlinearOperator, Operator
from .identity import IdentityOperator
from .linear import GeneralSVDOperator, LinearOperator, SVDOperator

__all__ = [
    "Operator",
    "NonlinearOperator",
    "IdentityOperator",
    "LinearOperator",
    "GeneralSVDOperator",
    "SVDOperator",
]
