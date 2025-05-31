from typing import Any

import torch
from diffusers import DDPMPipeline
from torch import Tensor

from samplers.dtypes import Device, DType
from samplers.networks import Network


class DDPMNetwork(Network):
    def __init__(self, pipeline: DDPMPipeline):

        acp = pipeline.scheduler.alphas_cumprod.clip(1e-6, 1)
        one = torch.tensor([1.0], dtype=acp.dtype, device=acp.device)
        alpha_cumprods = torch.cat([one, acp])

        super().__init__(alphas_cumprod=alpha_cumprods)
        self._model = pipeline.unet.eval().requires_grad_(False)

    # todo maybe do a base class with this inside to make it more dry put the pipeline type as class attribute
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        dtype: DType = None,
        device: Device = None,
        **pipeline_kwargs: Any,
    ) -> "DDPMNetwork":
        pipeline = DDPMPipeline.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            dtype=dtype,
            device=device,
            **pipeline_kwargs,
        )
        return cls(pipeline)

    def forward(self, sample: Tensor, t: Tensor | int) -> Tensor:  # noqa: N802
        return self._model(sample=sample, timestep=t).sample

    def set_timesteps(self, num_sampling_steps: int):
        timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long)
        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)
