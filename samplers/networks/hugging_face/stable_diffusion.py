import dataclasses
from typing import Any

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from samplers.dtypes import Shape
from samplers.networks import LatentNetwork


@dataclasses.dataclass(slots=True)
class StableDiffusionCondition:
    # textual guidance
    prompt: str | list[str] = ""
    negative_prompt: str | list[str] | None = None
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1

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
    classifier_free_guidance: bool
    guidance_scale: float
    cross_attention_kwargs: dict[str, Any] | None
    ip_adapter_embeds: torch.Tensor | None


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
        # todo also one idea could be to create mirror of diffusers "pipeline" and just have to give the name of the repository
        #  network = StableDiffusionNetwork(
        #  create from_pretrained
        betas = pipeline.scheduler.betas
        acp = (1.0 - betas).cumprod(dim=0)
        one = torch.tensor([1.0], dtype=betas.dtype, device=betas.device)
        alpha_cumprods = torch.cat([one, acp])

        super().__init__(alphas_cumprod=alpha_cumprods)
        self._pipeline: StableDiffusionPipeline = pipeline
        self._unet: UNet2DConditionModel = pipeline.unet.eval().requires_grad_(False)
        self._vae: AutoencoderKL = pipeline.vae.eval().requires_grad_(False)

        # todo find better names for those three
        # The multiplication factor of the latents
        self._latent_scaling_factor = pipeline.vae.config.scaling_factor
        # The size change of the latents
        self.latent_scale_factor = pipeline.vae_scale_factor
        self.num_latent_channels = pipeline.vae.config.latent_channels

        self._conditioning: ConditioningState | None = None

    def set_timesteps(self, num_sampling_steps: int):
        timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long)
        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)

    def get_latent_shape(self, x_shape: Shape) -> Shape:
        channel, height, width = x_shape

        scale_factor = self.latent_scale_factor
        if (height % scale_factor) or (width % scale_factor):
            raise ValueError(
                f"H={height} and W={width} must both be divisible by the latent_scale_factor={scale_factor}."
            )

        return self.num_latent_channels, height // scale_factor, width // scale_factor

    @torch.inference_mode()
    def set_condition(self, condition: StableDiffusionCondition | None) -> None:
        """Cache prompt / negative-prompt / image embeddings for the next
        batch.

        Parameters
        ----------
        condition : StableDiffusionCondition
            The conditioning specification created by the caller.
        """
        condition = condition if condition is not None else StableDiffusionCondition()

        # unpack once for readability
        cfg_active = condition.guidance_scale > 1.0
        lora_scale = (
            condition.cross_attention_kwargs.get("scale")
            if condition.cross_attention_kwargs
            else None
        )

        # 1. Text embeddings
        pos_embeds, neg_embeds = self._pipeline._encode_prompt(  # pyright: ignore
            condition.prompt,
            device=self.device,
            num_images_per_prompt=condition.num_images_per_prompt,
            do_classifier_free_guidance=cfg_active,
            negative_prompt=condition.negative_prompt,
            prompt_embeds=condition.prompt_embeds,
            negative_prompt_embeds=condition.negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=condition.clip_skip,
        )
        prompt_embeddings = torch.cat([neg_embeds, pos_embeds]) if cfg_active else pos_embeds

        # 2. IP-Adapter image embeddings
        image_embeds = None
        if condition.ip_adapter_image is not None or condition.ip_adapter_image_embeds is not None:
            image_embeds = self._pipeline.prepare_ip_adapter_image_embeds(
                condition.ip_adapter_image,
                condition.ip_adapter_image_embeds,
                device=self.device,
                batch_size=prompt_embeddings.size(0),
                do_classifier_free_guidance=cfg_active,
            )

        # 3. Store everything for forward()
        self._conditioning = ConditioningState(
            prompt_embeds=prompt_embeddings,
            classifier_free_guidance=cfg_active,
            guidance_scale=condition.guidance_scale,
            cross_attention_kwargs=condition.cross_attention_kwargs,
            ip_adapter_embeds=image_embeds,
        )

    def forward(self, latents: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """Return ε(xₜ, t) for the given latents and timestep."""
        if self._conditioning is None:
            raise RuntimeError("Call `set_condition()` before sampling.")

        state = self._conditioning

        latent_input = torch.cat([latents, latents]) if state.classifier_free_guidance else latents
        added_cond_kwargs = (
            {"image_embeds": state.ip_adapter_embeds}
            if state.ip_adapter_embeds is not None
            else None
        )

        # Main UNet call.
        eps = self._unet(
            sample=latent_input,
            timestep=t,
            encoder_hidden_states=state.prompt_embeds,
            cross_attention_kwargs=state.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Collapse back to a single batch using the CFG formula.
        if state.classifier_free_guidance:
            eps_uncond, eps_text = eps.chunk(2)
            eps = eps_uncond + state.guidance_scale * (eps_text - eps_uncond)

        return eps

    @torch.inference_mode()
    def _decode(self, z: torch.Tensor, *, differentiable: bool = True) -> torch.Tensor:
        """Convert latent z → image x using the VAE decoder."""
        z_scaled = z / self._latent_scaling_factor
        with torch.set_grad_enabled(differentiable):
            images = self._vae.decode(z_scaled, return_dict=False)[0]
        return images

    @torch.inference_mode()
    def _encode(self, x: torch.Tensor, *, differentiable: bool = True) -> torch.Tensor:
        """Convert image x → latent z using the VAE encoder (returns mean)."""
        with torch.set_grad_enabled(differentiable):
            distribution: DiagonalGaussianDistribution = self._vae.encode(x, return_dict=False)[0]
            z_scaled = distribution.mean * self._latent_scaling_factor
        return z_scaled

    @property
    def device(self) -> torch.device:  # noqa: D401
        """Device on which the adapter’s parameters live."""
        return self.alphas_cumprod.device

    def clear_condition(self) -> None:
        """Remove cached embeddings to release VRAM."""
        self._conditioning = None
