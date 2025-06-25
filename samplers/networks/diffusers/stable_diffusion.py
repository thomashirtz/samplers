import dataclasses
from typing import Any

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from samplers.dtypes import Device, DType, Shape

from ..base import DiffusionType, LatentEpsilonNetwork


@dataclasses.dataclass(slots=True)
class StableDiffusionCondition:
    # guidance
    prompt: str | list[str] = ""
    negative_prompt: str | list[str] | None = None
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0

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
    guidance_rescale: float
    cross_attention_kwargs: dict[str, Any] | None
    add_cond_kwargs: dict[str, Any] | None
    timestep_cond: torch.Tensor | None


class StableDiffusionNetwork(LatentEpsilonNetwork[StableDiffusionCondition]):
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
        acp = pipeline.scheduler.alphas_cumprod
        one = acp.new_tensor([1.0])
        alphas_cumprod = torch.cat([one, acp])
        super().__init__(alphas_cumprod=alphas_cumprod)

        self._conditioning: ConditioningState | None = None
        self._pipeline: StableDiffusionPipeline = pipeline
        self._pipeline.unet.eval().requires_grad_(False)
        self._pipeline.vae.eval().requires_grad_(False)

        # The factor by which the VAE multiplies its latent tensors
        self._vae_latent_multiplier = self.vae.config.scaling_factor
        # The ratio by which the latent’s spatial dimensions are scaled
        self.latent_resolution_ratio = self._pipeline.vae_scale_factor
        # The number of channels contained in each latent tensor
        self.latent_num_channels = self.vae.config.latent_channels

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

    def set_sampling_parameters(
        self,
        num_sampling_steps: int,
        batch_size: int = 1,
        num_reconstructions: int = 1,
        # height: int,
        # width: int,
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
        """Prepare and cache every tensor the UNet needs for the next denoising
        pass.

        The implementation purposefully mirrors
        :pymeth:`diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.__call__`:
        the only differences are that *self* has been replaced by the internal
        pipeline (``self._pipeline``) and that the many keyword arguments have
        been collected in a single :class:`~StableDiffusionCondition` object.

        Parameters
        ----------
        condition : StableDiffusionCondition | None
            A bundle of textual and/or image prompts as well as CFG parameters.
            If *None*, an empty ``StableDiffusionCondition()`` is used which
            yields unconditional generation.
        """
        if not self.are_sampling_parameters_initialized:
            raise RuntimeError("Call `set_sampling_parameters()` before conditioning.")

        condition = condition if condition is not None else StableDiffusionCondition()

        # 1. Check inputs. Raise error if not correct
        self._pipeline.check_inputs(
            prompt=condition.prompt,
            negative_prompt=condition.negative_prompt,
            prompt_embeds=condition.prompt_embeds,
            negative_prompt_embeds=condition.negative_prompt_embeds,
            ip_adapter_image=condition.ip_adapter_image,
            ip_adapter_image_embeds=condition.ip_adapter_image_embeds,
            # Height/width are checked in the original pipeline; here we pass
            # dummy values because the sampler controls the latent resolution.
            # In the future maybe height and width will be set (e.g. in
            # set_sampling_parameters or here) and checked.
            height=8,
            width=8,
            callback_steps=None,
        )

        self._pipeline._guidance_scale = (
            condition.guidance_scale
        )  # needed to call self._pipeline.do_classifier_free_guidance
        self._pipeline._guidance_rescale = condition.guidance_rescale
        self._pipeline._clip_skip = condition.clip_skip
        self._pipeline._cross_attention_kwargs = condition.cross_attention_kwargs
        self._pipeline._interrupt = False

        # 2. Define call parameters
        # todo there is actually an issue, the whole sampler is supposed to be able to run with batch_shape of any shape
        #  and everything seems to be good for this, except the handling of the prompt. (because we would need to handle
        #  list of list of prompt or something like this. If we really want to keep this flexibility, we can implement
        #  an utility to do that, otherwise we can simplify the sampler to accept only batch size and not batch shape.
        if condition.prompt is not None and isinstance(condition.prompt, str):
            batch_size = 1
        elif condition.prompt is not None and isinstance(condition.prompt, list):
            batch_size = len(condition.prompt)
        else:
            batch_size = condition.prompt_embeds.shape[0]

        if batch_size != self._batch_size:
            raise ValueError(
                "Batch size mismatch: received "
                f"{batch_size} prompt(s) but the sampler was initialised with "
                f"batch_size={self._batch_size}.\n\n"
                "How to fix this: either\n"
                " • supply exactly this number of prompts/embeddings, OR\n"
                " • call 'set_sampling_parameters' again with the desired batch_size.\n\n"
                "Hint: to generate multiple images in one go, pass a *list* of prompts "
                "— e.g. ['prompt-1', 'prompt-2', ...]."
            )
        num_images_per_prompt = self._num_reconstructions

        # In the future we might implement an `always_cfg` flag to force classic two-pass CFG even when time_cond_proj_dim ≠ None.
        # That would let us compare guidance strength on LCM-style UNets and avoid silent CFG disable when swapping models.
        do_classifier_free_guidance = self._pipeline.do_classifier_free_guidance

        # 3. Encode input prompt
        lora_scale = (
            self._pipeline.cross_attention_kwargs.get("scale", None)
            if self._pipeline.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self._pipeline.encode_prompt(
            prompt=condition.prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=condition.negative_prompt,
            prompt_embeds=condition.prompt_embeds,
            negative_prompt_embeds=condition.negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self._pipeline.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 2. IP-Adapter image embeddings
        if condition.ip_adapter_image is not None or condition.ip_adapter_image_embeds is not None:
            image_embeds = self._pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image=condition.ip_adapter_image,
                ip_adapter_image_embeds=condition.ip_adapter_image_embeds,
                device=self.device,
                num_images_per_prompt=batch_size * num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        # 6.1 Add image embeds for IP-Adapter
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
        if (
            self._pipeline.unet.config.time_cond_proj_dim is not None
        ):  # not sure which unet to choose here
            guidance_scale_tensor = torch.tensor(self._pipeline.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self._pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self._pipeline.unet.config.time_cond_proj_dim,  # not sure which unet to choose here
            ).to(device=self._pipeline.device, dtype=self._pipeline.dtype)

        # 3. Store everything for forward()
        self._conditioning = ConditioningState(
            prompt_embeds=prompt_embeds,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=condition.guidance_scale,
            cross_attention_kwargs=condition.cross_attention_kwargs,
            add_cond_kwargs=added_cond_kwargs,
            timestep_cond=timestep_cond,
            guidance_rescale=condition.guidance_rescale,
        )

    def forward(self, latents: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """Return ε(xₜ, t) for the given latents and timestep."""

        if not self.is_condition_initialized:
            raise RuntimeError("Call `set_condition()` before sampling.")

        state = self._conditioning

        latent_model_input = (
            torch.cat([latents] * 2) if state.do_classifier_free_guidance else latents
        )
        latent_model_input = self._pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Main UNet call.
        noise_pred = self.unet(
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

        if state.do_classifier_free_guidance and state.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(
                noise_pred, noise_pred_text, guidance_rescale=state.guidance_rescale
            )

        return noise_pred

    @torch.inference_mode()
    def _decode(self, z: torch.Tensor, *, differentiable: bool = False) -> torch.Tensor:
        """Convert latent z → image x using the VAE decoder."""
        z_scaled = z / self._vae_latent_multiplier
        with torch.set_grad_enabled(differentiable):
            images = self.vae.decode(z_scaled, return_dict=False)[0]
        return images

    @torch.inference_mode()
    def _encode(self, x: torch.Tensor, *, differentiable: bool = False) -> torch.Tensor:
        """Convert image x → latent z using the VAE encoder (returns mean)."""
        with torch.set_grad_enabled(differentiable):
            distribution: DiagonalGaussianDistribution = self.vae.encode(x, return_dict=False)[0]
            z_scaled = distribution.mean * self._vae_latent_multiplier
        return z_scaled

    @property
    def device(self) -> torch.device:  # noqa: D401
        """Device on which the adapter’s parameters live."""
        return self._pipeline.device

    @property
    def dtype(self) -> torch.dtype:  # noqa: D401
        """Device on which the adapter’s parameters live."""
        return self._pipeline.dtype

    @property
    def unet(self) -> UNet2DConditionModel:  # code elsewhere still uses self._unet
        return self._pipeline.unet

    @property
    def vae(self) -> AutoencoderKL:
        return self._pipeline.vae

    def clear_condition(self) -> None:
        """Remove cached embeddings to release VRAM."""
        self._conditioning = None
        torch.cuda.empty_cache()

    def is_condition_initialized(self):
        return self._conditioning is not None

    @property
    def diffusion_type(self) -> DiffusionType:
        return DiffusionType.VARIANCE_PRESERVING
