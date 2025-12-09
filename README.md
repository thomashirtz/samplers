# `samplers`

`samplers` makes **posterior sampling** with diffusion models almost plug-and-play.

Posterior sampling lets you tackle a broad class of **inverse problems** — such as denoising, inpainting, jpeg decompression, or deblurring — by combining two ingredients:

1. **A degradation operator** that describes how your data were corrupted.
2. **A strong generative prior** (any pretrained diffusion model, pixel or latent) that knows what clean signals look like.

Instead of training a new network for every task, you reuse an off-the-shelf model and draw samples from the posterior distribution `p(x | y)`. The result: you can solve problems the model was never trained for, often with state-of-the-art quality and zero extra training.

## Why you might want `samplers`

- **Bring-your-own model.** Works with many diffusion models from Hugging Face, easily extendable.
- **Operator + noise abstraction.** Drop-in classes for blurs, masks, Gaussian or Poisson noise, … or write your own in a few lines.
- **Pixel *or* latent space.** Pairs naturally with frameworks such as Stable Diffusion.
- **Typed, modular code.** Clear abstract hierarchy.
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
from samplers.operators import IdentityOperator
from samplers.samplers import DPSSampler
from samplers.utils.image import pil_to_tensor, tensor_to_pil

# 1. Instantiate a pretrained prior
model_name = "google/ddpm-celebahq-256"
network  = DDPMNetwork.from_pretrained(model_name)

# 2. Inverse problem construction
x_true = pil_to_tensor(Image.open("image.png"))  
operator = IdentityOperator(x_shape=x_true.shape)
noise = GaussianNoise(sigma=0.05)

# For the demonstration; in practice inverse problems start from y = A(x) + noise
problem = InverseProblem.from_clean_data(x_true, noise, operator)

# 3. Sample from the posterior
sampler = DPSSampler(network)
x_hat = sampler(problem)

tensor_to_pil(x_hat).save("restored.png")
```

## License

Released under the **BSD 3‑Clause License** © 2025 Thomas Hirtz.
See [`LICENSE`](LICENSE.md) for the full text.
