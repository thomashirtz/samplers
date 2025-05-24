import torch
from diffusers import DDPMPipeline

from samplers.config import MODELS_DIRECTORY
from samplers.inverse_problem import InverseProblem
from samplers.noise import GaussianNoise
from samplers.operators import IdentityOperator
from samplers.samplers import DPSSampler

if __name__ == "__main__":

    dtype = torch.bfloat16

    model_id = "google/ddpm-celebahq-256"
    ddpm_pipeline = DDPMPipeline.from_pretrained(model_id, cache_dir=MODELS_DIRECTORY, dtype=dtype)

    data = torch.ones((3, 128, 128), dtype=dtype)
    operator = IdentityOperator()
    noise = GaussianNoise(sigma=0.05)

    inverse_problem = InverseProblem.from_clean_data(
        x_true=data,
        noise=noise,
        operator=operator,
    )

    sampler = DPSSampler(model_or_pipeline=ddpm_pipeline)
    x_hat = sampler(inverse_problem=inverse_problem)
