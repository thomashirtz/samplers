import torch
from diffusers import AutoencoderKL, DDPMPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor

from samplers.networks import LatentNetwork, Network


class DDPMNetwork(Network):
    def __init__(self, pipeline: DDPMPipeline):

        acp = pipeline.scheduler.alphas_cumprod.clip(1e-6, 1)
        one = torch.tensor([1.0], dtype=acp.dtype, device=acp.device)
        alpha_cumprods = torch.cat([one, acp])

        super().__init__(alphas_cumprod=alpha_cumprods)
        self._model = pipeline.unet.eval().requires_grad_(False)

    def forward(self, sample: Tensor, t: Tensor | int) -> Tensor:  # noqa: N802
        return self._model(sample=sample, timestep=t).sample

    def set_timesteps(self, num_sampling_steps: int):
        timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long)
        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)


class StableDiffusionNetwork(LatentNetwork):
    def __init__(self, pipeline: StableDiffusionPipeline):
        dtype = pipeline.scheduler.betas.dtype
        device = pipeline.scheduler.betas.device
        acp = (1.0 - pipeline.scheduler.betas).cumprod(dim=0)
        one = torch.tensor([1.0], dtype=dtype, device=device)
        alpha_cumprods = torch.cat([one, acp])

        super().__init__(alphas_cumprod=alpha_cumprods)
        self._unet: UNet2DConditionModel = pipeline.unet.eval().requires_grad_(False)
        self._vae: AutoencoderKL = pipeline.vae.eval().requires_grad_(False)

    def forward(self, x: Tensor, t: Tensor | int, encoder_hidden_states) -> Tensor:
        # fixme how to handle the encoder hidden states (prompt, ...)
        return self._unet(sample=x, timestep=t, encoder_hidden_states=encoder_hidden_states).sample

    def set_timesteps(self, num_sampling_steps: int):
        timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long)
        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)

    def _decode(self, z: Tensor):
        return self._vae.decode(z=z, return_dict=False).sample

    def _encode(self, x: Tensor):
        distribution: DiagonalGaussianDistribution = self._vae.encode(x=x, return_dict=False)[0]
        return distribution.mean
