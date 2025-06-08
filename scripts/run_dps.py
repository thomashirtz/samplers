import torch
from PIL import Image

from samplers.config import MODELS_DIRECTORY
from samplers.inverse_problem import InverseProblem
from samplers.networks import DDPMNetwork
from samplers.noise import GaussianNoise
from samplers.operators import IdentityOperator
from samplers.samplers import DPSSampler
from samplers.utils.image import pil_to_tensor, tensor_to_pil

if __name__ == "__main__":

    dtype = torch.bfloat16
    device = "cuda:0"

    model_id = "google/ddpm-celebahq-256"
    network = DDPMNetwork.from_pretrained(
        model_id, cache_dir=MODELS_DIRECTORY, dtype=dtype, device=device
    )

    image = Image.open("ddpm.png")
    x_shape = pil_to_tensor(image).shape

    x_true = pil_to_tensor(image)
    # x_true = torch.stack((x_true, x_true))
    # x_true = torch.ones((3, 256, 256), dtype=dtype)

    operator = IdentityOperator(x_shape=x_shape)
    noise = GaussianNoise(sigma=0.05)

    inverse_problem = InverseProblem.from_clean_data(
        x_true=x_true.to(device=device, dtype=dtype),
        noise=noise,
        operator=operator,
    )

    sampler = DPSSampler(network=network)
    x_hat = sampler(inverse_problem=inverse_problem, num_sampling_steps=4)
    sample = tensor_to_pil(x_hat)
    sample.save("dps.jpg")
