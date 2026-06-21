"""Run ReSample on SD 1.5 for a noisy identity inverse problem.

Defaults match mgdm ``configs/experiments/sampler/resample.yaml``:
  nsteps=500, max_optimization_iters=200, sigma_scale=40, obs std=0.05

Usage:
  PYTHONPATH=. python scripts/run_resample.py
  PYTHONPATH=. python scripts/run_resample.py --quick   # 50 steps, faster smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from samplers.config import MODELS_DIRECTORY
from samplers.inverse_problem import InverseProblem
from samplers.networks import StableDiffusionCondition, StableDiffusionNetwork
from samplers.noise import GaussianNoise
from samplers.operators import IdentityOperator
from samplers.samplers.resample import ReSampleSampler
from samplers.utils.image import pil_to_tensor, tensor_to_pil

SCRIPT_DIR = Path(__file__).resolve().parent


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = (x - y).pow(2).mean().item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()


def load_image(path: Path, size: int, *, dtype: torch.dtype, device: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size and (image.width != size or image.height != size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return pil_to_tensor(image, dtype=dtype, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReSample + SD1.5 identity denoising demo")
    parser.add_argument("--image", type=Path, default=SCRIPT_DIR / "ddpm.png")
    parser.add_argument("--size", type=int, default=512, help="SD1.5 native resolution")
    parser.add_argument("--sigma", type=float, default=0.05, help="Observation noise (mgdm std)")
    parser.add_argument("--steps", type=int, default=None, help="Diffusion steps (default 500)")
    parser.add_argument("--opt-iters", type=int, default=None, help="AdamW iters (default 200)")
    parser.add_argument(
        "--quick", action="store_true", help="50 steps / 50 opt iters for a fast check"
    )
    args = parser.parse_args()

    if args.quick:
        num_steps = args.steps or 50
        max_opt = args.opt_iters or 50
    else:
        num_steps = args.steps or 500
        max_opt = args.opt_iters or 200

    dtype = torch.float32
    device = "cuda:0"

    print(
        f"ReSample demo: steps={num_steps}, opt_iters={max_opt}, sigma={args.sigma}, size={args.size}"
    )

    model_id = "sd-legacy/stable-diffusion-v1-5"
    network = StableDiffusionNetwork.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=MODELS_DIRECTORY, device=device
    )

    x_true = load_image(args.image, args.size, dtype=dtype, device=device)
    x_shape = x_true.shape

    operator = IdentityOperator(x_shape=x_shape)
    noise = GaussianNoise(sigma=args.sigma, device=device, dtype=dtype)

    inverse_problem = InverseProblem.from_clean_data(
        x_true=x_true,
        noise=noise,
        operator=operator,
    )

    condition = StableDiffusionCondition(prompt="", guidance_scale=1.0)

    sampler = ReSampleSampler(network=network)
    x_hat = sampler(
        inverse_problem=inverse_problem,
        num_sampling_steps=num_steps,
        num_reconstructions=1,
        max_optimization_iters=max_opt,
        sigma_scale=40.0,
        eta=1.0,
        condition=condition,
    )
    x_hat = x_hat[0]

    x_obs = inverse_problem.observation

    ref_pil = tensor_to_pil(x_true)
    obs_pil = tensor_to_pil(x_obs)
    rec_pil = tensor_to_pil(x_hat)

    ref_pil.save(SCRIPT_DIR / "resample_reference.png")
    obs_pil.save(SCRIPT_DIR / "resample_observation.png")
    rec_pil.save(SCRIPT_DIR / "resample_reconstruction.png")

    psnr_ref = psnr(x_hat.detach().float(), x_true.float())
    psnr_obs = psnr(x_hat.detach().float(), x_obs.float())
    print(f"PSNR(reconstruction, reference): {psnr_ref:.2f} dB")
    print(f"PSNR(reconstruction, observation): {psnr_obs:.2f} dB")
    print(f"PSNR(observation, reference): {psnr(x_obs.float(), x_true.float()):.2f} dB")
    print(f"Saved outputs to {SCRIPT_DIR}/resample_*.png")
