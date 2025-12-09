from abc import ABC

from samplers.models import CustomModel
from samplers.networks import EpsilonNetwork


class CustomEpsilonNetwork(EpsilonNetwork, ABC):
    def __init__(self, model: CustomModel):
        model = model.requires_grad_(False)
        self.model = model.eval()
        super().__init__(alphas_cumprod=model.alpha_cumprods, timesteps=model.timesteps)
