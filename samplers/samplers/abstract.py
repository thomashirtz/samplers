from abc import ABC

from samplers.networks.utilities import _build_network


class PosteriorSampler(ABC):
    def __init__(self, model_or_pipeline):
        self.network = _build_network(model_or_pipeline)

    def __call__(
        self,
        num_sampling_steps: int = 1,
        *args,
        **kwargs,
    ):
        pass
