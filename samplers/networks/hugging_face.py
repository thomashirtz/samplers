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


from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.utils import PipelineImageInput


# --------------------------------------------------------------------------- #
# 1.  Batch-level conditioning cache                                          #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(slots=True)
class ConditioningState:
    """Container holding every tensor needed by the UNet during a sampling
    run."""

    prompt_embeds: torch.Tensor
    classifier_free_guidance: bool
    guidance_scale: float
    cross_attention_kwargs: Optional[Dict[str, Any]]
    ip_adapter_embeds: Optional[torch.Tensor]


# --------------------------------------------------------------------------- #
# 2.  Base class providing α-cumprod utilities                                #
# --------------------------------------------------------------------------- #
class LatentNetwork(torch.nn.Module):
    """Abstract ε-network that knows the cumulative‐product schedule ᾱₜ."""

    def __init__(self, alphas_cumprod: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("timesteps", tensor=None, persistent=False)

    def _predict_x0(
        self, noisy_latents: torch.Tensor, t: torch.Tensor | int, eps: torch.Tensor
    ) -> torch.Tensor:
        r"""Recover x₀ from xₜ and ε according to DDPM posterior."""
        alpha_bar_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return (noisy_latents - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)


# --------------------------------------------------------------------------- #
# 3.  Stable-Diffusion adapter                                                #
# --------------------------------------------------------------------------- #
class SDNetworkAdapter(LatentNetwork):
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

    # --------------------------------------------------------------------- #
    # Construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(self, pipeline: StableDiffusionPipeline) -> None:
        self.pipeline: StableDiffusionPipeline = pipeline

        betas = pipeline.scheduler.betas
        alpha_cumprod = torch.cat(
            [torch.ones(1, dtype=betas.dtype, device=betas.device), (1.0 - betas).cumprod(dim=0)]
        )
        super().__init__(alpha_cumprod)

        self._unet: UNet2DConditionModel = pipeline.unet.eval().requires_grad_(False)
        self._vae = pipeline.vae.eval().requires_grad_(False)

        self._conditioning: Optional[ConditioningState] = None

    # --------------------------------------------------------------------- #
    # Static configuration                                                  #
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def set_timesteps(self, num_steps: int) -> None:
        """Create an evenly spaced time grid in the DDPM domain [0 … 999]."""
        timesteps = torch.linspace(0, 999, steps=num_steps, dtype=torch.long, device=self.device)
        self.register_buffer("timesteps", timesteps, persistent=True)

    # --------------------------------------------------------------------- #
    # Per-batch conditioning                                                #
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def set_condition(
        self,
        *,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] | None = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        clip_skip: int | None = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: List[torch.Tensor] | None = None,
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
        pos_embeds, neg_embeds = self.pipeline._encode_prompt(  # pyright: ignore
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
            image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
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

    # --------------------------------------------------------------------- #
    # ε-network forward pass                                                #
    # --------------------------------------------------------------------- #
    def forward(self, latents: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """Return ε(xₜ, t) for the given latents and timestep."""
        if self._conditioning is None:
            raise RuntimeError("Call `set_condition()` before sampling.")

        state = self._conditioning

        # Duplicate latents when classifier-free guidance is active.
        latent_input = torch.cat([latents, latents]) if state.classifier_free_guidance else latents

        # Main UNet call.
        eps = self._unet(
            sample=latent_input,
            timestep=t,
            encoder_hidden_states=state.prompt_embeds,
            cross_attention_kwargs=state.cross_attention_kwargs,
            added_cond_kwargs=(
                {"image_embeds": state.ip_adapter_embeds}
                if state.ip_adapter_embeds is not None
                else None
            ),
        ).sample

        # Collapse back to a single batch using the CFG formula.
        if state.classifier_free_guidance:
            eps_uncond, eps_text = eps.chunk(2)
            eps = eps_uncond + state.guidance_scale * (eps_text - eps_uncond)

        return eps

    # --------------------------------------------------------------------- #
    # Convenience wrappers                                                  #
    # --------------------------------------------------------------------- #
    def predict_x0(self, latents: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """Recover denoised latents x₀ given noisy latents xₜ and timestep
        t."""
        return self._predict_x0(latents, t, self.forward(latents, t))

    # --- latent-space ↔ pixel-space helpers -------------------------------- #
    @torch.inference_mode()
    def _decode(self, z: torch.Tensor, *, differentiable: bool = True) -> torch.Tensor:
        """Convert latent z → image x using the VAE decoder."""
        with torch.set_grad_enabled(differentiable):
            decoded = self._vae.decode(z, return_dict=False)[0]
        return decoded

    @torch.inference_mode()
    def _encode(self, x: torch.Tensor, *, differentiable: bool = True) -> torch.Tensor:
        """Convert image x → latent z using the VAE encoder (returns mean)."""
        with torch.set_grad_enabled(differentiable):
            distribution = self._vae.encode(x, return_dict=False)[0]
        return distribution.mean

    # --- misc -------------------------------------------------------------- #
    @property
    def device(self) -> torch.device:  # noqa: D401
        """Device on which the adapter’s parameters live."""
        return self.alphas_cumprod.device

    def clear_condition(self) -> None:
        """Remove cached embeddings to release VRAM."""
        self._conditioning = None
