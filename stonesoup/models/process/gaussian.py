from datetime import timedelta
from typing import List

import numpy as np
from scipy.linalg import inv

from stonesoup.types.track import Track
from ...kernel import Kernel, SquaredExponentialKernel
from ...base import Base, Property


class GaussianProcess(Base):
    lag: int = Property(default=3, doc="Length of the sliding window")
    sigma: float = Property(default=1.0, doc="process noise")
    kernel: SquaredExponentialKernel = Property(doc="kernel")

    def _calc_kernel_matrix(self, timestamps: List[timedelta]):

        n = len(timestamps)
        k = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k[i, j] = self.kernel(timestamps[i], timestamps[j])
        return k

    def matrix(self, track: Track, time_interval: timedelta, **kwargs):
        timestamps = self._get_timestamps(track, time_interval)

        # Calculate kernel matrix
        k = self._calc_kernel_matrix(timestamps)

        # Make notation analogous to Kalman Filter
        p_xy = np.atleast_2d(k[0, 1:])
        p_yy = np.atleast_2d(k[1:, 1:])
        inv_p_yy = inv(p_yy) if len(p_yy) else p_yy

        # Calculate A
        n = len(timestamps)
        row_1 = np.concatenate((p_xy @ inv_p_yy, np.zeros((1, self.lag - n + 1))), axis=1)
        row_2 = np.concatenate((np.eye(self.lag - 1), np.zeros((self.lag - 1, 1))),
                               axis=1)
        A = np.concatenate((row_1, row_2))
        return A
