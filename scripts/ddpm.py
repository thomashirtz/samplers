import torch
from diffusers import DDPMPipeline

from samplers.config import MODELS_DIRECTORY

if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.bfloat16
    model_id = "google/ddpm-celebahq-256"
    ddpm = DDPMPipeline.from_pretrained(
        model_id, cache_dir=MODELS_DIRECTORY, dtype=dtype, device=device
    )
    image = ddpm(num_inference_steps=200)["images"]
    image[0].save("ddpm.png")
