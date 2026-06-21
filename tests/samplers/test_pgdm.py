import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

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
NoCondition = _base.NoCondition

_networks_pkg = types.ModuleType("samplers.networks")
_networks_pkg.__path__ = [str(_root / "samplers" / "networks")]
_networks_pkg.EpsilonNetwork = EpsilonNetwork
_networks_pkg.LatentEpsilonNetwork = _base.LatentEpsilonNetwork
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
_utils_pkg.batch_view = _batch_view
_utils_pkg.bridge_kernels = _bridge
sys.modules["samplers.samplers.utils"] = _utils_pkg
sys.modules["samplers.samplers.utils.batch_view"] = _batch_view
sys.modules["samplers.samplers.utils.bridge_kernels"] = _bridge

_pgdm = _load_module("samplers/samplers/pgdm.py", "pgdm")
PGDMSampler = _pgdm.PGDMSampler


class MockEpsilonNetwork(EpsilonNetwork[NoCondition]):
    """Minimal VP network: zero noise prediction → predict_x0 equals denoising formula on x."""

    def __init__(self, num_steps: int = 5) -> None:
        # Pad index 0 → 1.0; indices 1..num_steps use a simple alpha schedule.
        alphas = torch.linspace(0.99, 0.5, num_steps)
        alphas_cumprod = torch.cat([torch.tensor([1.0]), alphas])
        super().__init__(alphas_cumprod=alphas_cumprod)
        self._num_steps = num_steps

    def forward(self, x: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        return torch.zeros_like(x)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    def set_sampling_parameters(
        self,
        num_sampling_steps: int,
        batch_size: int = 1,
        num_reconstructions: int = 1,
    ) -> None:
        self._batch_size = batch_size
        self._num_sampling_steps = num_sampling_steps
        self._num_reconstructions = num_reconstructions
        # Ascending buffer indices (as after flip in diffusers adapters).
        timesteps = torch.arange(1, num_sampling_steps + 1, dtype=torch.long)
        self.register_buffer("timesteps", timesteps, persistent=True)

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


def test_pgdm_smoke_single_batch():
    x_shape = (2, 4)
    problem = _make_problem(batch_shape=(), x_shape=x_shape)
    sampler = PGDMSampler(MockEpsilonNetwork(num_steps=5))

    out = sampler(problem, num_sampling_steps=5, num_reconstructions=1, guidance_weight=0.1)

    assert out.shape == x_shape


def test_pgdm_batched_observations():
    x_shape = (2, 4)
    problem = _make_problem(batch_shape=(2,), x_shape=x_shape)
    sampler = PGDMSampler(MockEpsilonNetwork(num_steps=5))

    out = sampler(problem, num_sampling_steps=5, num_reconstructions=2, guidance_weight=0.1)

    assert out.shape == (2, 2, *x_shape)


def test_pgdm_ddim_step_uses_x0_pred():
    x_shape = (2, 4)
    problem = _make_problem(batch_shape=(), x_shape=x_shape)
    net = MockEpsilonNetwork(num_steps=5)
    sampler = PGDMSampler(net)
    calls: list[dict] = []

    def spy_ddim(*, x, epsilon_net, t, t_prev, eta, e_t=None, **kwargs):
        calls.append({"e_t": e_t.detach().clone(), "x": x.detach().clone(), "t": t})
        from samplers.samplers.utils.bridge_kernels import ddim_step as real_ddim

        return real_ddim(
            x=x, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t, **kwargs
        )

    with patch.object(_pgdm, "ddim_step", side_effect=spy_ddim):
        sampler(problem, num_sampling_steps=5, num_reconstructions=1, guidance_weight=0.0)

    assert calls, "ddim_step should be called at least once"
    call = calls[0]
    x0 = net.predict_x0(call["x"], call["t"])
    noise = net.predict_noise(call["x"], call["t"])
    assert torch.allclose(call["e_t"], x0)
    assert not torch.allclose(call["e_t"], noise)
