import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from samplers.inverse_problem import InverseProblem
from samplers.noise import GaussianNoise
from samplers.operators.identity import IdentityOperator

_root = Path(__file__).resolve().parents[2]


def _load_module(relative: str, name: str):
    path = _root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_module("samplers/networks/base.py", "networks_base")
EpsilonNetwork = _base.EpsilonNetwork
LatentEpsilonNetwork = _base.LatentEpsilonNetwork
NoCondition = _base.NoCondition

_networks_pkg = types.ModuleType("samplers.networks")
_networks_pkg.__path__ = [str(_root / "samplers" / "networks")]
_networks_pkg.EpsilonNetwork = EpsilonNetwork
_networks_pkg.LatentEpsilonNetwork = LatentEpsilonNetwork
_networks_pkg.base = _base
sys.modules["samplers.networks"] = _networks_pkg
sys.modules["samplers.networks.base"] = _base

_sampler_base = _load_module("samplers/samplers/base.py", "sampler_base")
_samplers_pkg = types.ModuleType("samplers.samplers")
_samplers_pkg.__path__ = [str(_root / "samplers" / "samplers")]
_samplers_pkg.PosteriorSampler = _sampler_base.PosteriorSampler
_samplers_pkg.base = _sampler_base
sys.modules["samplers.samplers"] = _samplers_pkg
sys.modules["samplers.samplers.base"] = _sampler_base

_utils_pkg = types.ModuleType("samplers.samplers.utils")
_utils_pkg.__path__ = [str(_root / "samplers" / "samplers" / "utils")]
_batch_view = _load_module("samplers/samplers/utils/batch_view.py", "batch_view")
_bridge = _load_module("samplers/samplers/utils/bridge_kernels.py", "bridge_kernels")
_resample_kernels = _load_module("samplers/samplers/utils/resample_kernels.py", "resample_kernels")
_utils_pkg.batch_view = _batch_view
_utils_pkg.bridge_kernels = _bridge
_utils_pkg.resample_kernels = _resample_kernels
sys.modules["samplers.samplers.utils"] = _utils_pkg
sys.modules["samplers.samplers.utils.batch_view"] = _batch_view
sys.modules["samplers.samplers.utils.bridge_kernels"] = _bridge
sys.modules["samplers.samplers.utils.resample_kernels"] = _resample_kernels

_resample = _load_module("samplers/samplers/resample.py", "resample")
ReSampleSampler = _resample.ReSampleSampler


class MockLatentEpsilonNetwork(LatentEpsilonNetwork[NoCondition]):
    """Minimal latent VP network with identity encode/decode."""

    def __init__(self, num_steps: int = 5) -> None:
        alphas = torch.linspace(0.99, 0.5, num_steps)
        alphas_cumprod = torch.cat([torch.tensor([1.0]), alphas])
        super().__init__(alphas_cumprod=alphas_cumprod)
        self._num_steps = num_steps

    def forward(self, x: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        return torch.zeros_like(x)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    def get_latent_shape(self, x_shape: tuple[int, ...]) -> tuple[int, ...]:
        return x_shape

    def _encode(self, x: torch.Tensor, *, differentiable: bool = False) -> torch.Tensor:
        return x

    def _decode(self, z: torch.Tensor, *, differentiable: bool = False) -> torch.Tensor:
        return z

    def set_sampling_parameters(
        self,
        num_sampling_steps: int,
        batch_size: int = 1,
        num_reconstructions: int = 1,
    ) -> None:
        self._batch_size = batch_size
        self._num_sampling_steps = num_sampling_steps
        self._num_reconstructions = num_reconstructions
        timesteps = torch.arange(1, num_sampling_steps + 1, dtype=torch.long)
        self.register_buffer("timesteps", timesteps, persistent=True)

    @property
    def is_condition_initialized(self) -> bool:
        return True

    def set_condition(self, condition: NoCondition | None = None) -> None:
        pass

    def clear_condition(self) -> None:
        pass


class PlainEpsilonNetwork(EpsilonNetwork[NoCondition]):
    def __init__(self) -> None:
        super().__init__(alphas_cumprod=torch.tensor([1.0, 0.9]))

    def forward(self, x: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        return torch.zeros_like(x)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    def set_sampling_parameters(self, num_sampling_steps: int, batch_size: int = 1, **_) -> None:
        self.register_buffer("timesteps", torch.arange(1, num_sampling_steps + 1), persistent=True)

    @property
    def is_condition_initialized(self) -> bool:
        return True

    def set_condition(self, condition: NoCondition | None = None) -> None:
        pass

    def clear_condition(self) -> None:
        pass


def _make_problem(*, batch_shape: tuple[int, ...], x_shape: tuple[int, ...]) -> InverseProblem:
    operator = IdentityOperator(x_shape=x_shape)
    observation = torch.randn(*batch_shape, *x_shape)
    noise = GaussianNoise(sigma=0.01)
    return InverseProblem(operator=operator, observation=observation, noise=noise)


def _fast_kwargs(**overrides):
    defaults = dict(
        num_sampling_steps=5,
        max_optimization_iters=5,
        time_travel_interval=100,
    )
    defaults.update(overrides)
    return defaults


def test_resample_smoke_single_batch():
    x_shape = (2, 4)
    problem = _make_problem(batch_shape=(), x_shape=x_shape)
    sampler = ReSampleSampler(MockLatentEpsilonNetwork(num_steps=5))

    out = sampler(problem, num_reconstructions=1, **_fast_kwargs())

    assert out.shape == (1, *x_shape)


def test_resample_batched_observations():
    x_shape = (2, 4)
    problem = _make_problem(batch_shape=(2,), x_shape=x_shape)
    sampler = ReSampleSampler(MockLatentEpsilonNetwork(num_steps=5))

    out = sampler(problem, num_reconstructions=2, **_fast_kwargs())

    assert out.shape == (2, 2, *x_shape)


def test_resample_decode_output_false():
    x_shape = (2, 4)
    problem = _make_problem(batch_shape=(), x_shape=x_shape)
    sampler = ReSampleSampler(MockLatentEpsilonNetwork(num_steps=5))

    out = sampler(problem, decode_output=False, **_fast_kwargs())

    assert out.shape == (1, *x_shape)


def test_resample_requires_latent_network():
    with pytest.raises(TypeError, match="latent diffusion model"):
        ReSampleSampler(PlainEpsilonNetwork())


def test_stochastic_resample_shape():
    b, c, h, w = 2, 3, 4, 4
    pseudo_x0 = torch.randn(b, c, h, w)
    x_t = torch.randn(b, c, h, w)
    a_t = torch.full((b, 1, 1, 1), 0.8)
    sigma = torch.full((b, 1, 1, 1), 0.5)

    out = _resample_kernels.stochastic_resample(pseudo_x0, x_t, a_t, sigma)

    assert out.shape == pseudo_x0.shape
