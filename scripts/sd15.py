import torch
from diffusers import StableDiffusionPipeline

from samplers.config import MODELS_DIRECTORY

# from samplers.networks import StableDiffusionNetwork
from samplers.utils.image import pil_to_tensor

if __name__ == "__main__":
    dtype = torch.float16
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=MODELS_DIRECTORY,
    )
    pipe = pipe.to("cuda")
    batch_size = 2
    num_images_per_prompt = 2
    prompt = ["a photo of an astronaut riding a horse on mars"] * batch_size
    # prompt = "a photo of an astronaut riding a horse on mars"

    image = pipe(
        prompt, num_inference_steps=2, num_images_per_prompt=num_images_per_prompt, guidance_scale=8
    ).images[0]

    image.save("astronaut_rides_horse.png")
    data = pil_to_tensor(image=image).unsqueeze(0).to(dtype).to("cuda")
    # n = StableDiffusionNetwork(pipeline=pipe)
    # a = n.encode(data)
    # b = n.forward(a, 1)
    # print()

    # bs 1, nipp 1, guidance scale = 7.5 => torch.Size([2, 4, 64, 64]) (noise)
    # bs 1, nipp 2, guidance scale = 7.5 => torch.Size([4, 4, 64, 64])
    # bs 1, nipp 2, guidance scale = 0   => torch.Size([2, 4, 64, 64])
    # bs 2, nipp 2, guidance scale = 0   => torch.Size([4, 4, 64, 64])
    # bs 2, nipp 2, guidance scale = 0   => torch.Size([4, 4, 64, 64])
    # bs 2, nipp 2, guidance scale = 8   => torch.Size([8, 4, 64, 64])
