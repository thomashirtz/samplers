import dataclasses
from typing import Any

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from samplers.networks import LatentNetwork


@dataclasses.dataclass(slots=True)
class ConditioningState:
    """Container holding every tensor needed by the UNet during a sampling
    run."""

    prompt_embeds: torch.Tensor
    classifier_free_guidance: bool
    guidance_scale: float
    cross_attention_kwargs: dict[str, Any] | None
    ip_adapter_embeds: torch.Tensor | None


class SDNetwork(LatentNetwork):
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
        acp = (1.0 - betas).cumprod(dim=0)
        one = torch.tensor([1.0], dtype=betas.dtype, device=betas.device)
        alpha_cumprods = torch.cat([one, acp])

        super().__init__(alphas_cumprod=alpha_cumprods)
        self._pipeline: StableDiffusionPipeline = pipeline
        self._unet: UNet2DConditionModel = pipeline.unet.eval().requires_grad_(False)
        self._vae: AutoencoderKL = pipeline.vae.eval().requires_grad_(False)
        self._latent_scaling_factor = pipeline.vae.config.scaling_factor

        self._conditioning: ConditioningState | None = None

    def set_timesteps(self, num_sampling_steps: int):
        timesteps = torch.linspace(start=0, end=999, steps=num_sampling_steps, dtype=torch.long)
        self.register_buffer(name="timesteps", tensor=timesteps, persistent=True)

    @torch.inference_mode()
    def set_condition(
        self,
        *,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,  # todo find out how to handle that
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        clip_skip: int | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: list[torch.Tensor] | None = None,
    ) -> None:
        """Compute and store all tensors required by the UNet for one batch.

        This method **must** be called every time the prompt batch changes.

        Parameters
        ----------
        prompt
            Text prompt(s) that guide generation.
        negative_prompt
            Negative prompt(s) used for classifier-free guidance.
        guidance_scale
            Strength of classifier-free guidance.  Values > 1 enable CFG.
        num_images_per_prompt
            How many images will be generated for *each* prompt string.
        prompt_embeds, negative_prompt_embeds
            Pre-computed embeddings (skip CLIP encode step if supplied).
        clip_skip
            Number of CLIP layers to skip (see SDXL paper).
        cross_attention_kwargs
            Extra kwargs forwarded to all `CrossAttention` modules (e.g. LoRA).
        ip_adapter_image, ip_adapter_image_embeds
            Optional conditioning images or their embeddings.
        """
        cfg_active = guidance_scale > 1.0

        # 1. Text embeddings via Diffusers’ internal helper
        lora_scale = cross_attention_kwargs.get("scale") if cross_attention_kwargs else None
        pos_embeds, neg_embeds = self._pipeline._encode_prompt(  # pyright: ignore
            prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=cfg_active,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )
        prompt_embeddings = torch.cat([neg_embeds, pos_embeds]) if cfg_active else pos_embeds

        # 2. Image embeddings for IP-Adapter (if requested)
        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self._pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device=self.device,
                batch_size=prompt_embeddings.size(0),
                do_classifier_free_guidance=cfg_active,
            )

        # 3. Store everything inside a dataclass for easy inspection
        self._conditioning = ConditioningState(
            prompt_embeds=prompt_embeddings,
            classifier_free_guidance=cfg_active,
            guidance_scale=guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
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
