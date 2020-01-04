import numpy as np
import numba, scipy
from numba import njit, prange


class KDE_Numba:

    """Kernel Density Estimation with Numba
    Call KDE_Numba like you would stats.gaussian_kde"""

    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1/sum(self._weights**2)

        self.set_bandwidth(bw_method=bw_method)


    def evaluate(self, points):
        return self.kde_numba(points, np.squeeze(self.dataset), 
                              self.weights, self.bw)

    __call__ = evaluate


    @staticmethod
    @njit(parallel=True, cache=True)
    def kde_numba(x_eval, x_sampl, weights, bw):
        n = x_eval.shape[0]
        exps = np.zeros(n, dtype=np.float64)
        for i in prange(n):
            p = x_eval[i]
            d = (-(p-x_sampl)**2)/(2*bw**2)
            exps[i] += np.sum(np.exp(d)*weights)
        fac = np.sqrt(2*np.pi)*bw
        return exps/fac


    def scotts_factor(self):
        return np.power(self.neff, -1./(self.d+4))


    def silverman_factor(self):
        return np.power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))


    def set_bandwidth(self, bw_method=None):
        if bw_method is None:
            self.covariance_factor = self.scotts_factor
        elif bw_method == "scott":
            self.covariance_factor = self.scotts_factor
        elif bw_method == "silverman":
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self.covariance_factor = lambda: bw_method
        else:
            msg = "`bw_method` should be 'scott', 'silverman', or a scalar."
            raise ValueError(msg) 
        self.rescale_bandwidth()


    def rescale_bandwidth(self):
        cov = scipy._lib._numpy_compat.cov(self.dataset, 
                                           rowvar=1,
                                           bias=False,
                                           aweights=self.weights)
        self.bw = np.sqrt(cov)*self.covariance_factor()


    def pdf(self, x):
        return self.evaluate(x)


    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = np.ones(self.n)/self.n
            return self._weights


    @property
    def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1/sum(self.weights**2)
            return self._neff


