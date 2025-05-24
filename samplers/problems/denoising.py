from .base import InverseProblem
from .operators import IdentityOperator


class Denoising(InverseProblem):
    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(shape)
        self.operator = IdentityOperator(input_shape=shape)

    def __call__(self, data):
        pass
