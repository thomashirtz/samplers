from diffusers import DDPMPipeline, StableDiffusionPipeline

from samplers.models import CustomModel
from samplers.networks import CustomNetwork, DDPMNetwork, Network, StableDiffusionNetwork


def build_network(model_or_pipeline) -> Network:
    if isinstance(model_or_pipeline, DDPMPipeline):
        return DDPMNetwork(model_or_pipeline)
    if isinstance(model_or_pipeline, CustomModel):
        return CustomNetwork(model_or_pipeline)
    if isinstance(model_or_pipeline, StableDiffusionPipeline):
        return StableDiffusionNetwork(model_or_pipeline)
    else:
        raise NotImplementedError(model_or_pipeline)
