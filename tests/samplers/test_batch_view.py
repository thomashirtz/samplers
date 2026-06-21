import importlib.util
from pathlib import Path

# Import directly to avoid pulling diffusers via samplers.samplers.__init__
_module_path = (
    Path(__file__).resolve().parents[2] / "samplers" / "samplers" / "utils" / "batch_view.py"
)
_spec = importlib.util.spec_from_file_location("batch_view", _module_path)
assert _spec and _spec.loader
_batch_view = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch_view)
BatchView = _batch_view.BatchView


def test_shape_property():
    view = BatchView(batch_shape=(2, 3), num_samples=4, data_shape=(3, 64, 64))
    assert view.shape == (2, 3, 4, 3, 64, 64)


def test_shape_scalar_batch():
    view = BatchView(batch_shape=(), num_samples=2, data_shape=(1, 8, 8))
    assert view.shape == (2, 1, 8, 8)
