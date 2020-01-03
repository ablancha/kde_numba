# kde\_numba

A simple implementation of Kernel Density Estimation with Numba. 

`KDE_Numba` behaves exactly like `scipy.stats.gaussian_kde` (same syntax, same arguments). For large arrays, Numba provides a speed-up of one to two orders of magnitude compared to Scipy.

![](/src/perf.svg)


## Dependencies

* [numpy](http://www.numpy.org/)
* [numba](http://numba.pydata.org) 
* [matplotlib](https://matplotlib.org) for plotting
* [scipy](https://www.scipy.org) for comparison
* [perfplot](https://pypi.org/project/perfplot/) for benchmarking


Send comments and questions to ablancha@mit.edu
