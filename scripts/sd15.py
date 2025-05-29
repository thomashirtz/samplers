import torch
from diffusers import StableDiffusionPipeline

from samplers.config import MODELS_DIRECTORY

if __name__ == "__main__":
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=MODELS_DIRECTORY,
    )
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
    image.save("astronaut_rides_horse.png")
