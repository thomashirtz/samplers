import torch
from diffusers import DDPMPipeline
from torch import Tensor

from samplers.networks import Network


class HuggingFaceNetwork(Network):
    def __init__(self, pipeline: DDPMPipeline, num_sampling_steps: int = 100):
        timesteps = torch.linspace(0, 999, num_sampling_steps, dtype=torch.long)

        acp = pipeline.scheduler.alphas_cumprod.clip(1e-6, 1)
        one = torch.tensor([1.0], dtype=acp.dtype, device=acp.device)
        alpha_cumprods = torch.cat([one, acp])

        super().__init__(alpha_cumprods=alpha_cumprods, timesteps=timesteps)

        self._model = pipeline.unet.eval().requires_grad_(False)

    def model(self, x: Tensor, t: Tensor) -> Tensor:  # noqa: N802
        return self._model(x, t)
