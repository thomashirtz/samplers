import dataclasses
from typing import Any

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from samplers.dtypes import Device, DType, Shape
from samplers.networks import LatentNetwork


@dataclasses.dataclass(slots=True)
class StableDiffusionCondition:
    # textual guidance
    prompt: str | list[str] = ""
    negative_prompt: str | list[str] | None = None
    guidance_scale: float = 7.5
    # num_images_per_prompt: int = 1

    # optional pre-computed embeddings
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    clip_skip: int | None = None

    # cross-attention / LoRA tweaks
    cross_attention_kwargs: dict[str, Any] | None = None

    # IP-Adapter support
    ip_adapter_image: PipelineImageInput | None = None
    ip_adapter_image_embeds: list[torch.Tensor] | None = None


@dataclasses.dataclass(slots=True)
class ConditioningState:
    """Container holding every tensor needed by the UNet during a sampling
    run."""

    prompt_embeds: torch.Tensor
    do_classifier_free_guidance: bool
    guidance_scale: float
    cross_attention_kwargs: dict[str, Any] | None
    add_cond_kwargs: dict[str, Any] | None
    timestep_cond: torch.Tensor | None


class StableDiffusionNetwork(LatentNetwork[StableDiffusionCondition]):
    """Expose a `StableDiffusionPipeline` as an ε-network for posterior
    samplers.

    The class reuses as much code as possible from Diffusers, so future
    improvements to tokenisation, LoRA, ControlNet, etc. are inherited
    automatically.

    Public methods
    --------------
    set_timesteps
        Define the discrete time grid used by the sampler.
    set_condition
        Cache prompt / negative-prompt / image embeddings for the next batch.
    forward
        Predict ε(xₜ, t).  Called inside the sampler’s denoising loop.
    predict_x0
        Convenience wrapper that returns the denoised latents x₀.
    _encode / _decode
        VAE helpers for switching between pixel and latent space.
    """

    def __init__(self, pipeline: StableDiffusionPipeline) -> None:
        betas = pipeline.scheduler.betas
        acp = (1.0 - betas).cumprod(dim=0).to(dtype=pipeline.dtype, device=pipeline.device)
        one = torch.tensor([1.0], dtype=pipeline.dtype, device=pipeline.device)
        alpha_cumprods = torch.cat([one, acp])

        super().__init__(alphas_cumprod=alpha_cumprods)

        self._conditioning: ConditioningState | None = None
        self._pipeline: StableDiffusionPipeline = pipeline
        self._unet: UNet2DConditionModel = pipeline.unet.eval().requires_grad_(False)
        self._vae: AutoencoderKL = pipeline.vae.eval().requires_grad_(False)

        # The factor by which the VAE multiplies its latent tensors
        self._vae_latent_multiplier = pipeline.vae.config.scaling_factor
        # The ratio by which the latent’s spatial dimensions are scaled
        self.latent_resolution_ratio = pipeline.vae_scale_factor
        # The number of channels contained in each latent tensor
        self.latent_num_channels = pipeline.vae.config.latent_channels

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        torch_dtype: DType = None,
        device: Device = None,
        **pipeline_kwargs: Any,
    ) -> "StableDiffusionNetwork":
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            **pipeline_kwargs,
        )
        pipeline = pipeline.to(device)
        return cls(pipeline=pipeline)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        # _pipeline is not a nn.Module, therefore anything below _pipeline won't be move by the default .to.
        # The .to has therefore been extended to move the pipeline as well.
        device_kwargs = dict(device=device) if device else {}
        self._pipeline = self._pipeline.to(dtype=dtype, **device_kwargs)
        super().to(dtype=dtype, **device_kwargs)
        return self

    def set_timesteps(self, num_sampling_steps: int):
        timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long)
        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)
        # CGPT said that SDXL is between 0-1111, it proposes:
        # self._pipeline.scheduler.set_timesteps(num_sampling_steps, device=self.device)
        # self.register_buffer("timesteps",
        #                      self._pipeline.scheduler.timesteps.to(torch.long),
        #                      persistent=True)
        # todo need to check if it works
        # Hard-coding 0‒999 still breaks SDXL (0‒1111) and any custom scheduler.
        # The two-liner you commented out is the robust solution – just remember that many schedulers keep the dtype
        # as float, so don’t cast to long unless your sampler insists on it.

    def get_latent_shape(self, x_shape: Shape) -> Shape:
        channel, height, width = x_shape

        scale_factor = self.latent_resolution_ratio
        if (height % scale_factor) or (width % scale_factor):
            raise ValueError(
                f"H={height} and W={width} must both be divisible by the latent_scale_factor={scale_factor}."
            )

        return self.latent_num_channels, height // scale_factor, width // scale_factor

    @torch.inference_mode()
    def set_condition(self, condition: StableDiffusionCondition | None) -> None:
        # todo add num_reconstructions and batch size to the set_condition
        """Cache prompt / negative-prompt / image embeddings for the next
        batch. See the __call__ methods of StableDiffusionPipeline for the
        details.

        Parameters
        ----------
        condition : StableDiffusionCondition
            The conditioning specification created by the caller.
        """
        condition = condition if condition is not None else StableDiffusionCondition()

        # fixme number of sample in the original code:
        #  if prompt is not None and isinstance(prompt, str):
        #      batch_size = 1
        #  elif prompt is not None and isinstance(prompt, list):
        #      batch_size = len(prompt)
        #  else:
        #      batch_size = prompt_embeds.shape[0]

        batch_size = 1  # todo add args
        num_images_per_prompt = 1

        # unpack once for readability
        do_classifier_free_guidance = condition.guidance_scale > 1.0
        lora_scale = (
            condition.cross_attention_kwargs.get("scale")
            if condition.cross_attention_kwargs
            else None
        )

        # 1. Text embeddings
        prompt_embeddings, negative_prompt_embeddings = self._pipeline.encode_prompt(
            prompt=condition.prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=condition.negative_prompt,
            prompt_embeds=condition.prompt_embeds,
            negative_prompt_embeds=condition.negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=condition.clip_skip,
        )
        if do_classifier_free_guidance:
            prompt_embeddings = torch.cat([negative_prompt_embeddings, prompt_embeddings])

        # 2. IP-Adapter image embeddings
        image_embeds = None
        if condition.ip_adapter_image is not None or condition.ip_adapter_image_embeds is not None:
            image_embeds = self._pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image=condition.ip_adapter_image,
                ip_adapter_image_embeds=condition.ip_adapter_image_embeds,
                device=self.device,
                num_images_per_prompt=batch_size * num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (
                condition.ip_adapter_image is not None
                or condition.ip_adapter_image_embeds is not None
            )
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self._unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(condition.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self._pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self._unet.config.time_cond_proj_dim
            ).to(device=self.device, dtype=self._unet.dtype)

        # 3. Store everything for forward()
        self._conditioning = ConditioningState(
            prompt_embeds=prompt_embeddings,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=condition.guidance_scale,
            cross_attention_kwargs=condition.cross_attention_kwargs,
            add_cond_kwargs=added_cond_kwargs,
            timestep_cond=timestep_cond,
        )

    def forward(self, latents: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """Return ε(xₜ, t) for the given latents and timestep."""
        if self._conditioning is None:
            raise RuntimeError("Call `set_condition()` before sampling.")

        state = self._conditioning

        latent_model_input = (
            torch.cat([latents] * 2) if state.do_classifier_free_guidance else latents
        )
        latent_model_input = self._pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Main UNet call.
        noise_pred = self._unet(
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=state.prompt_embeds,
            cross_attention_kwargs=state.cross_attention_kwargs,
            added_cond_kwargs=state.add_cond_kwargs,
            timestep_cond=state.timestep_cond,
        ).sample

        # Collapse back to a single batch using the CFG formula.
        if state.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + state.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if state.do_classifier_free_guidance and self._pipeline.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(
                noise_pred, noise_pred_text, guidance_rescale=self._pipeline.guidance_rescale
            )

        return noise_pred

    @torch.inference_mode()
    def _decode(self, z: torch.Tensor, *, differentiable: bool = True) -> torch.Tensor:
        """Convert latent z → image x using the VAE decoder."""
        z_scaled = z / self._vae_latent_multiplier
        with torch.set_grad_enabled(differentiable):
            images = self._vae.decode(z_scaled, return_dict=False)[0]
        return images

    @torch.inference_mode()
    def _encode(self, x: torch.Tensor, *, differentiable: bool = True) -> torch.Tensor:
        """Convert image x → latent z using the VAE encoder (returns mean)."""
        with torch.set_grad_enabled(differentiable):
            distribution: DiagonalGaussianDistribution = self._vae.encode(x, return_dict=False)[0]
            z_scaled = distribution.mean * self._vae_latent_multiplier
        return z_scaled

    @property
    def device(self) -> torch.device:  # noqa: D401
        """Device on which the adapter’s parameters live."""
        return self.alphas_cumprod.device

    def clear_condition(self) -> None:
        """Remove cached embeddings to release VRAM."""
        self._conditioning = None
        torch.cuda.empty_cache()

    # CGPT suggestions:
    def enable_sequential_cpu_offload(self, **kwargs):
        self._pipeline.enable_sequential_cpu_offload(**kwargs)
        return self

    def enable_xformers_memory_efficient_attention(self, **kwargs):
        self._pipeline.enable_xformers_memory_efficient_attention(**kwargs)
        return self
