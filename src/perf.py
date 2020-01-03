import numpy as np
from scipy import stats
import perfplot
from matplotlib import pyplot as plt
from kdenumba import KDE_Numba

plt.figure(figsize=(6,5))

for ii in range(1,5):

    def pdf_scipy(x):
        smpl = np.random.gumbel(0, 0.1, 10**ii)
        weights = np.ones(smpl.shape)
        return stats.gaussian_kde(smpl, weights=weights)(x)

    def pdf_numba(x):
        smpl = np.random.gumbel(0, 0.1, 10**ii)
        weights = np.ones(smpl.shape)
        return KDE_Numba(smpl, weights=weights)(x)

    plt.subplot(2,2,ii)
    out = perfplot.bench(
          setup=np.random.rand,
          n_range=[10**k for k in range(1,7)],
          kernels=[pdf_numba, pdf_scipy],
          xlabel='Length of x',
          labels=['Numba', 'Scipy'],
          title='Number of samples: $10^' + str(ii) + '$',
          equality_check=None
          )
    out.plot()

plt.tight_layout()
plt.savefig('perf.png')
plt.show()

