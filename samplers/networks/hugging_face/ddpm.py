from typing import Any

import torch
from diffusers import DDPMPipeline
from torch import Tensor

from samplers.dtypes import Device, DType

from ..base import Network, NoCondition


class DDPMNetwork(Network[NoCondition]):
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
        if self._num_sampling_steps is None:
            raise RuntimeError("Call `set_sampling_parameters()` before sampling.")
        return self._model(sample=sample, timestep=t).sample

    def set_sampling_parameters(
        self,
        num_sampling_steps: int,
        batch_size: int = 1,
        num_reconstructions: int = 1,
    ):
        self._batch_size = batch_size
        self._num_sampling_steps = num_sampling_steps
        self._num_reconstructions = num_reconstructions
        timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long)
        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)
