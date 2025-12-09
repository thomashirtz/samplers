import torch
from PIL import Image

from samplers.config import MODELS_DIRECTORY
from samplers.inverse_problem import InverseProblem
from samplers.networks import StableDiffusionCondition, StableDiffusionNetwork
from samplers.noise import GaussianNoise
from samplers.operators import IdentityOperator
from samplers.samplers import PSLDSampler
from samplers.utils.image import pil_to_tensor, tensor_to_pil

if __name__ == "__main__":

    dtype = torch.bfloat16
    device = "cuda:0"

    model_id = "sd-legacy/stable-diffusion-v1-5"
    network = StableDiffusionNetwork.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=MODELS_DIRECTORY, device=device
    )

    image = Image.open("ddpm.png")
    x_shape = pil_to_tensor(image).shape
    batch_size = 2
    x_true = pil_to_tensor(image)
    x_true = torch.stack((x_true,) * batch_size).to(device=device, dtype=dtype)
    # x_true = torch.ones((3, 256, 256), dtype=dtype)

    operator = IdentityOperator(x_shape=x_shape)
    noise = GaussianNoise(sigma=0.05)

    inverse_problem = InverseProblem.from_clean_data(
        x_true=x_true.to(device=device, dtype=dtype),
        noise=noise,
        operator=operator,
    )
    condition = StableDiffusionCondition(prompt=[""] * batch_size)

    sampler = PSLDSampler(network=network)
    x_hat = sampler(
        inverse_problem=inverse_problem,
        num_sampling_steps=100,
        num_reconstructions=2,
        condition=condition,
    )
    sample = tensor_to_pil(x_hat[0, 0])
    sample.save("psld.jpg")
