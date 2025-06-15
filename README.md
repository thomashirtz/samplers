# `samplers`

`samplers` makes **posterior sampling** with diffusion models almost plug-and-play.

Posterior sampling lets you tackle a broad class of **inverse problems** — such as denoising, inpainting, jpeg decompression, or deblurring — by combining two ingredients:

1. **A degradation operator** that describes how your data were corrupted.
2. **A strong generative prior** (any pretrained diffusion model, pixel or latent) that knows what clean signals look like.

Instead of training a new network for every task, you reuse an off-the-shelf model and draw samples from the posterior distribution *p(x | y)*. The result: you can solve problems the model was never trained for, often with state-of-the-art quality and zero extra training.

## Why you might want `samplers`

- **Bring-your-own model.** Works with many diffusion models from Hugging Face, and easily create custom networks.
- **Operator + noise abstraction.** Drop-in classes for blurs, masks, sensing matrices, Gaussian or Poisson noise, … or write your own in a few lines.
- **Pixel *or* latent space.** Pairs naturally with Stable Diffusion + VAE pipelines.
- **Typed, modular code.** Clear ABC hierarchy, easy to extend.
- **Lean dependency stack.** PyTorch ≥ 2.0, `diffusers`, `torchvision` — nothing exotic.
- **Growing algorithm zoo.** State-of-the-art samplers already included, new ones coming out soon.

## Installation

```
git clone https://github.com/thomashirtz/samplers
cd samplers
pip install -e .
```

## Quick start — Image restoration in 16 lines

```python
from PIL import Image

from samplers.inverse_problem import InverseProblem
from samplers.networks import DDPMNetwork
from samplers.noise import GaussianNoise
from samplers.operators import IdentityOperator  # swap for other operators
from samplers.samplers import DPSSampler  # one of several ready-made samplers
from samplers.utils.image import pil_to_tensor, tensor_to_pil

# 1. Pretrained diffusion prior
model_name = "google/ddpm-celebahq-256"
network  = DDPMNetwork.from_pretrained(model_name)

# 2. Ground-truth image (for the demo; in practice you’d start from y = Ax + n)
x_true = pil_to_tensor(Image.open("image.png"))

# 3. Forward model and noise
operator = IdentityOperator(x_shape=x_true.shape)
noise = GaussianNoise(sigma=0.05)

problem = InverseProblem.from_clean_data(x_true, noise, operator)

# 4. Sample from the posterior
sampler = DPSSampler(network)
x_hat = sampler(problem)

tensor_to_pil(x_hat).save("restored.png")
```

## License

Released under the **BSD 3‑Clause License** © 2025 Thomas Hirtz.
See [`LICENSE`](LICENSE.md) for the full text.
