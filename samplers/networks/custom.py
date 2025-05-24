from samplers.models.custom import CustomModel
from samplers.networks.base import Network


class CustomNetwork(Network):
    def __init__(self, model: CustomModel):
        model = model.requires_grad_(False)
        self.model = model.eval()
        super().__init__(alpha_cumprods=model.alpha_cumprods, timesteps=model.timesteps)
