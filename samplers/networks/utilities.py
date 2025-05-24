from diffusers import DDPMPipeline

from samplers.models.custom import CustomModel
from samplers.networks.base import Network
from samplers.networks.custom import CustomNetwork
from samplers.networks.hugging_face import HuggingFaceNetwork


def _build_network(model_or_pipeline) -> Network:
    if isinstance(model_or_pipeline, DDPMPipeline):
        return HuggingFaceNetwork(model_or_pipeline)
    if isinstance(model_or_pipeline, CustomModel):
        return CustomNetwork(model_or_pipeline)
    else:
        raise NotImplementedError(model_or_pipeline)
