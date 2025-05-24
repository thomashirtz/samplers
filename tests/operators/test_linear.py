import pytest
import torch
from torch.testing import assert_close

from samplers.problems.operators.linear import GeneralSVDOperator


@pytest.fixture(scope="module")
def random_op():
    """Return a GeneralSVDOperator built from a random matrix once per
    module."""
    torch.manual_seed(0)
    H = torch.randn(12, 7)
    U, s, Vh = torch.linalg.svd(H, full_matrices=False)
    return GeneralSVDOperator(U, s, Vh)


def _dense_H(op):
    # Reconstruct H from factors once, reuse in all tests
    # return (op.U * op.s).T @ op.Vh            # (m, n)
    return (op._U * op._singular_values.unsqueeze(0)) @ op._V_transpose  # shape (m, n)


def test_apply(random_op):
    op = random_op
    x = torch.randn(5, op.shape[1])
    assert_close(op.apply(x), x @ _dense_H(op).T)


def test_transpose(random_op):
    op = random_op
    y = torch.randn(5, op.shape[0])
    assert_close(op.apply_transpose(y), y @ _dense_H(op))


def test_pinv(random_op):
    op = random_op
    y = torch.randn(5, op.shape[0])
    H_pinv = torch.linalg.pinv(_dense_H(op))
    assert_close(op.apply_pseudo_inverse(y), y @ H_pinv.T)


# todo see how it handles batches
