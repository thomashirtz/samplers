# samplers


*Samplers* is a small, self‑contained Python toolbox for posterior sampling with diffusion‑type networks.
Out of the box it includes DPS (Chung et al.,2022), PSLD (Rout et al.,2024) and flexible building blocks for new algorithms.

More posterior sampling algorithms are underway, stay tuned.

## Features

- Clean, torch-native API – bring your own diffusion model, operator and noise law.
- Works in pixel or latent space (Stable Diffusion, VAE, …).
- Minimal dependencies (PyTorch ≥ 2.0, diffusers, torchvision).

## Installation

```
git clone https://github.com/thomashirtz/samplers
cd samplers
pip install -e .
```

## Quick‑start (DPS on a single face image)

```python
from PIL import Image

from samplers.inverse_problem import InverseProblem
from samplers.networks import DDPMNetwork
from samplers.noise import GaussianNoise
from samplers.operators import IdentityOperator
from samplers.samplers import DPSSampler
from samplers.utils.image import pil_to_tensor, tensor_to_pil

model_id = "google/ddpm-celebahq-256"
network = DDPMNetwork.from_pretrained(model_id)

image = Image.open("image.png")
x_true = pil_to_tensor(image)

operator = IdentityOperator(x_shape=x_true.shape)
noise = GaussianNoise(sigma=0.05)

inverse_problem = InverseProblem.from_clean_data(
    x_true=x_true,
    noise=noise,
    operator=operator,
)

sampler = DPSSampler(network=network)
x_hat = sampler(inverse_problem=inverse_problem)
sample = tensor_to_pil(x_hat)
sample.save("dps.png")
```

## License

Released under the **BSD 3‑Clause License** © 2025 Thomas Hirtz.
See [`LICENSE`](LICENSE.md) for the full text.
