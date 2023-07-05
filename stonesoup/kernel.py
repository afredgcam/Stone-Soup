from abc import abstractmethod

import numpy as np

from .base import Base, Property


class Kernel(Base):
    """Kernel base type

    A Kernel provides a means to translate state space or measurement space into kernel space.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, state1, state2):
        r"""
        Compute the distance between a pair of :class:`~.State` objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            distance measure between a pair of input :class:`~.State` objects

        """
        return NotImplementedError


class SquaredExponentialKernel(Kernel):
    l: float = Property(doc="Length scale")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, state1, state2, *args, **kwargs):
        dt = state2.timestamp - state1.timestamp
        return np.exp(-dt.total_seconds() ** 2 / (2 * self.l ** 2))
