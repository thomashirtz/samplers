import math
from typing import Sequence, Tuple

import torch
from torch import Tensor


class BatchView:
    """Manage tensor shapes for batched sampling operations.

    This helper reshapes between two views of a tensor:

        Structured view:  (*batch_shape, num_samples, *data_shape)
        Flattened view:   (leading_size,          *data_shape)

    Definitions
    -----------
    batch_shape       tuple[int, ...]
        Leading dimensions of the observation batch.
    num_samples       int
        Number of posterior samples per observation.
    data_shape        tuple[int, ...]
        Dimensions of a single data item.
    leading_shape     tuple[int, ...]
        (*batch_shape, num_samples) — shape before flattening.
    leading_size      int
        Product of leading_shape — size of flattened leading axis.
    flat_shape        tuple[int, ...]
        (leading_size, *data_shape) — shape after flattening.

    Use `flatten` and `unflatten` to switch between views, and
    `repeat_observation` to tile the observation across samples before flattening.
    """

    def __init__(
        self,
        batch_shape: int | Sequence[int] | torch.Size,
        num_samples: int,
        data_shape: int | Sequence[int] | torch.Size,
    ) -> None:
        self._batch_shape: Tuple[int, ...] = self._to_tuple(batch_shape)
        self._num_samples: int = num_samples
        self._data_shape: Tuple[int, ...] = self._to_tuple(data_shape)

        self._leading_shape: Tuple[int, ...] = (*self._batch_shape, self._num_samples)
        self._leading_size: int = math.prod(self._leading_shape)

    @staticmethod
    def _to_tuple(x: int | Sequence[int] | torch.Size) -> Tuple[int, ...]:
        """Convert a scalar or iterable into a tuple of ints."""
        if isinstance(x, int):
            return (x,)
        return tuple(x)

    # --- Read-only properties -----------------------------------------

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """batch_shape — original observation batch dimensions."""
        return self._batch_shape

    @property
    def batch_size(self) -> int:
        """batch_size — product of batch_shape."""
        return math.prod(self._batch_shape)

    @property
    def num_samples(self) -> int:
        """num_samples — number of samples per observation."""
        return self._num_samples

    @property
    def data_shape(self) -> Tuple[int, ...]:
        """data_shape — dimensions of a single data item."""
        return self._data_shape

    @property
    def leading_shape(self) -> Tuple[int, ...]:
        """leading_shape — (*batch_shape, num_samples) before flattening."""
        return self._leading_shape

    @property
    def leading_size(self) -> int:
        """leading_size — size of the flattened leading axis."""
        return self._leading_size

    @property
    def flat_shape(self) -> Tuple[int, ...]:
        """flat_shape — (leading_size, *data_shape) after flattening."""
        return self._leading_size, *self._data_shape

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape — (*batch_shape, num_samples, *data_shape)."""
        return *self._leading_shape, self._num_samples, *self._data_shape

    @property
    def per_sample_broadcast_shape(self) -> Tuple[int, ...]:
        """Shape for per-sample statistics that must broadcast over
        spatial/channel axes:

        ``(leading_size, 1, 1, …)``.
        """
        return (self.leading_size,) + (1,) * len(self.data_shape)

    # --- Reshape helpers ----------------------------------------------

    def flatten(self, x: Tensor) -> Tensor:
        """Collapse the leading axes into one.

        Input shape:  (*batch_shape, num_samples, *tail)
        Output shape: (leading_size, *tail)
        """
        tail_ndim = len(self._data_shape)
        return x.reshape(self._leading_size, *x.shape[-tail_ndim:])

    def unflatten(self, x: Tensor) -> Tensor:
        """Restore the flattened view back to structured form.

        Input shape:  (leading_size, *tail)
        Output shape: (*batch_shape, num_samples, *tail)
        """
        tail_ndim = len(self._data_shape)
        return x.reshape(*self._leading_shape, *x.shape[-tail_ndim:])

    # --- Observation convenience -------------------------------------

    def repeat_observation(self, observation: Tensor) -> Tensor:
        """Tile an observation across the sample axis, then flatten.

        Input shape:  (*batch_shape, *data_shape)
        Output shape: (leading_size, *data_shape)
        """
        expanded = observation.unsqueeze(len(self._batch_shape)).expand(
            *self._leading_shape, *observation.shape[-len(self._data_shape) :]
        )
        return self.flatten(expanded)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"batch_shape={self._batch_shape}, "
            f"num_samples={self._num_samples}, "
            f"data_shape={self._data_shape})"
        )
