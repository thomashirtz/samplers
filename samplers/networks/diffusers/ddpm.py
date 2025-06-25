from typing import Any

import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline, UNet2DModel
from torch import Tensor

from samplers.dtypes import Device, DType

from ..base import DiffusionType, EpsilonNetwork, NoCondition


class DDPMNetwork(EpsilonNetwork[NoCondition]):
    def __init__(self, pipeline: DDPMPipeline):
        acp = pipeline.scheduler.alphas_cumprod
        one = acp.new_tensor([1.0])
        alphas_cumprod = torch.cat([one, acp])
        super().__init__(alphas_cumprod=alphas_cumprod)
        self._conditioning: NoCondition | None = None
        self._pipeline: DDPMPipeline = pipeline
        self._pipeline.unet.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        torch_dtype: DType = None,
        device: Device = None,
        **pipeline_kwargs: Any,
    ) -> "DDPMNetwork":
        pipeline = DDPMPipeline.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            **pipeline_kwargs,
        )
        pipeline = pipeline.to(device)
        return cls(pipeline)

    def forward(self, sample: Tensor, t: Tensor | int) -> Tensor:  # noqa: N802
        if self._num_sampling_steps is None:
            raise RuntimeError("Call `set_sampling_parameters()` before sampling.")
        return self.unet(sample=sample, timestep=t).sample

    def set_sampling_parameters(
        self,
        num_sampling_steps: int,
        batch_size: int = 1,
        num_reconstructions: int = 1,
    ):
        self._batch_size = batch_size
        self._num_sampling_steps = num_sampling_steps
        self._num_reconstructions = num_reconstructions

        self._pipeline.scheduler.set_timesteps(num_sampling_steps, device=self.device)
        timesteps = torch.flip(self._pipeline.scheduler.timesteps, dims=(0,))
        # todo investigate the effects of using the sampler's timesteps instead of the ones from the pipeline.
        #  We need to see if including the borns change something. (pipeline 0-999, sampler 0-990)
        #  timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long, device=self.device)

        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)

    def is_condition_initialized(self) -> bool:
        """Unconditional model."""
        return True

    @property
    def unet(self) -> UNet2DModel:  # code elsewhere still uses self._unet
        return self._pipeline.unet

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = torch.device(device) if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype
        super().to(device=device, dtype=dtype)
        self._pipeline = self._pipeline.to(device=device, dtype=dtype)
        return self

    @property
    def device(self) -> torch.device:  # noqa: D401
        """Device on which the adapter’s parameters live."""
        return self._pipeline.device

    @property
    def dtype(self) -> torch.dtype:  # noqa: D401
        """Device on which the adapter’s parameters live."""
        return self._pipeline.dtype

    @property
    def diffusion_type(self) -> DiffusionType:
        return DiffusionType.VARIANCE_PRESERVING
