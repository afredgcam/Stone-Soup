import copy
from datetime import timedelta

import numpy as np
from numpy.linalg import solve

from stonesoup.types.track import Track
from ..base import Property
from ..models.process.gaussian import GaussianProcess
from ..predictor import Predictor
from ..types.prediction import Prediction


class GaussianProcessPredictor(Predictor):
    transition_model: GaussianProcess = Property(doc="Gaussian process")

    def predict(self, prior, track: Track, timestamp=None, **kwargs):
        dt = timedelta(seconds=1)
        x_pred = self.transition_model.matrix(track, time_interval=dt, **kwargs) @ prior.state_vector
        ct = copy.copy(prior.covar)
        ct_star = ct[1:, 1:]
        ckt_star = ct[-1, :-1]
        c_inv_c = solve(ct_star, np.atleast_2d(ckt_star).T).flatten()

        print(ct_star)
        print(ckt_star)
        print(c_inv_c, type(c_inv_c))
        p_pred = np.zeros([self.transition_model.lag, self.transition_model.lag])
        p_pred[:-1,:-1] = ct_star
        p_pred[:-1, -1] = c_inv_c
        p_pred[-1, :-1] = c_inv_c
        p_pred[-1, -1] = \
            np.atleast_2d(ct_star[-1, -1]) - np.atleast_2d(ckt_star) @ np.atleast_2d(c_inv_c).T

        return Prediction.from_state(prior, x_pred, p_pred, timestamp=timestamp,
                                     transition_model=self.transition_model)
