import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from ...base import Property, Base


class PoissonMixIn(Base):
    lambda_0: float = Property()
    lambda_1: float = Property()
    area: float = Property()

    def logpdf(self, state1, state2, **kwargs):
        return np.sum([
            logsumexp([
                np.log(self.lambda_1) + multivariate_normal.logpdf((meas.state_vector - self.function(state2, **kwargs)).T,
                                                                    cov=self.covar(**kwargs)),
                np.full((state2.state_vector.shape[1],), np.log(self.lambda_0) - np.log(self.area)),
                ], axis=0)
            for meas in state1],
            axis=0)