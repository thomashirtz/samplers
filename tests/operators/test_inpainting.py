# test_inpainting_ops.py
import pytest
import torch

from samplers.operators.inpainting import SidePaintingOperator  # ← must exist
from samplers.operators.inpainting import (  # ← adjust import path
    CenterInpaintingOperator,
    CenterOutpaintingOperator,
    InpaintingOperator,
    get_mask_inpaint_center,
    get_mask_side_painting,
)


def _assert_roundtrip(op: InpaintingOperator, x: torch.Tensor):
    kept = op.apply_V_transpose(x)
    recon = op.apply_V(kept)
    mask = op.mask

    assert torch.allclose(recon[~mask], x[~mask])
    assert torch.all(recon[mask] == 0)

    m, n = op.shape
    assert m == (~mask).sum().item()
    assert n == mask.numel()
    assert op.get_singular_values().numel() == m


@pytest.mark.parametrize("batch,c,h,w", [(2, 3, 4, 4), (1, 1, 5, 7)])
def test_center_inpaint_roundtrip(batch, c, h, w):
    x = torch.randn(batch, c, h, w)
    op = CenterInpaintingOperator((c, h, w), paint_fraction=0.5)
    _assert_roundtrip(op, x)


@pytest.mark.parametrize("batch,c,h,w", [(2, 3, 4, 4), (1, 1, 6, 6)])
def test_center_outpaint_roundtrip(batch, c, h, w):
    x = torch.randn(batch, c, h, w)
    op = CenterOutpaintingOperator((c, h, w), keep_fraction=0.5)
    _assert_roundtrip(op, x)


@pytest.mark.parametrize("left", [True, False])
def test_side_inpaint_roundtrip(left):
    batch, c, h, w = 2, 3, 4, 8
    x = torch.randn(batch, c, h, w)
    op = SidePaintingOperator((c, h, w), paint_fraction=0.5, left=left)
    _assert_roundtrip(op, x)


def test_manual_mask_constructor():
    x_shape = (3, 4, 4)
    mask = get_mask_inpaint_center(x_shape, 0.25, 0.75)
    op = InpaintingOperator(x_shape, mask)
    m, n = op.shape
    assert m + mask.sum().item() == n


cuda = torch.cuda.is_available()


@pytest.mark.skipif(not cuda, reason="CUDA not available")
def test_device_follows_mask_by_default():
    x_shape = (1, 4, 4)
    mask_gpu = get_mask_side_painting(x_shape, 0.5, True, device="cuda")
    op = InpaintingOperator(x_shape, mask_gpu)
    assert op.mask.device.type == "cuda"


@pytest.mark.skipif(not cuda, reason="CUDA not available")
def test_override_device_argument():
    x_shape = (1, 4, 4)
    mask_gpu = get_mask_side_painting(x_shape, 0.5, True, device="cuda")
    op = InpaintingOperator(x_shape, mask_gpu, device="cpu")
    assert op.mask.device.type == "cpu"
