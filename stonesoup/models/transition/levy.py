import datetime

import numpy as np

from .linear import CombinedLinearGaussianTransitionModel
from ...types.array import StateVectors
from ...base import Property


class StableLevyCombinedModel(CombinedLinearGaussianTransitionModel):
    alpha: float = Property(doc="Alpha")
    jumps: int = Property(doc="Number of jumps")

    def rvs(self, num_samples=1, time_interval=None):
        dt = time_interval.total_seconds()

        def h_stable(alpha):
            gamma_sequence = np.random.exponential(scale=1, size=self.jumps).cumsum()
            return gamma_sequence ** (-1 / alpha)

        I = np.zeros((self.ndim, num_samples))

        for p in range(num_samples):

            x_series = h_stable(self.alpha) * (dt ** (1 / self.alpha))

            V = np.random.uniform(low=0., high=dt, size=x_series.size)
            U = np.random.normal(loc=0., scale=1., size=(x_series.size, len(self.model_list)))

            for i in range(V.size):
                n_offset = 0
                for j, model in enumerate(self.model_list):
                    eA = model.matrix(time_interval=time_interval - datetime.timedelta(seconds=V[i]))
                    h = np.zeros((model.ndim, 1))
                    h[model.ndim - 1, 0] = 1
                    eAh = eA @ h
                    eAhUX = eAh * np.sqrt(model.noise_diff_coeff) * U[i, j] * x_series[i]
                    I[n_offset:n_offset + model.ndim, p] += eAhUX.ravel()
                    n_offset += model.ndim
        return StateVectors(dt ** (1 / self.alpha) * I)

    def logpdf(self, state1, state2):
        pass