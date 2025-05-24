from abc import ABC, abstractmethod

from samplers.networks import build_network


class PosteriorSampler(ABC):
    def __init__(self, model_or_pipeline):
        self._epsilon_network = build_network(model_or_pipeline)
