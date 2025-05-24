from abc import ABC

from .operators import Operator


class InverseProblem(ABC):
    operator: Operator
    noise: float
