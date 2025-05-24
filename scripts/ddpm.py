from diffusers import DDPMPipeline

from samplers.config import MODELS_DIRECTORY

if __name__ == "__main__":

    model_id = "google/ddpm-celebahq-256"
    ddpm = DDPMPipeline.from_pretrained(model_id, cache_dir=MODELS_DIRECTORY)
    image = ddpm()["sample"]
    image[0].save("ddpm_generated_image.png")
