from datetime import timedelta, datetime

import numpy as np

from ..process import GaussianProcessPredictor
from ...kernel import SquaredExponentialKernel
from ...models.process.gaussian import GaussianProcess
from ...types.array import StateVector
from ...types.state import GaussianState, State
from ...types.track import Track


def test_predictor():

    gp = GaussianProcess(kernel=SquaredExponentialKernel(l=1))
    predictor = GaussianProcessPredictor(gp)

    state = GaussianState([1, 0, 0], covar=np.array([[1, np.exp(-1), np.exp(-4)],
                                                     [np.exp(-1), 1, np.exp(-1)],
                                                     [np.exp(-4), np.exp(-1), 1]]))
    track = Track([State(StateVector([1, 0, 0]), timestamp=datetime.now()+timedelta(seconds=0)),
                   State(StateVector([1, 0, 0]), timestamp=datetime.now() + timedelta(seconds=1)),
                    ])
    prediction = predictor.predict(state, track, timestamp=timedelta(seconds=1))

    true_mean = np.array([[1], [1], [1]])
    true_covar = np.array([[1, np.exp(-1), (np.exp(-4) - np.exp(-2)) / (1 - np.exp(-2))],
              [np.exp(-1), 1, (np.exp(-1) - np.exp(-5)) / (1 - np.exp(-2))],
              [(np.exp(-4) - np.exp(-2)) / (1 - np.exp(-2)),
               (np.exp(-1) - np.exp(-5)) / (1 - np.exp(-2)),
               1 - (np.exp(-8) - 2 * np.exp(-6) + np.exp(-2)) / (1 - np.exp(-2))]])

    assert (np.allclose(prediction.covar, true_covar))
    assert (np.allclose(prediction.mean, true_mean))
