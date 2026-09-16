#
# reynspec.py  -- Reynolds Spectra fitting
#
#
# Using MCMC on CUDA GPU through the Python class imgpu.Mcgpu.
#

import imgpu
import numpy as np
import time
import sys
from numpy import pi, array, float32, int32, uint8, int64, uint64, where, \
     zeros, ones, arange, ones_like, linspace, sqrt, argsort
from matplotlib.pyplot import figure, plot, subplot, show, hist, grid, axis, \
	xlabel, ylabel, title

ndim = 6
varnames = ['theta0','lambda1','T0', 'psi','r0', 'ymax']


vars = np.array([
    [0.05, 0.879,   1.55],
    [1e13, 1.28e13, 3.7e13],
    [9000,  9836, 110000],
    [1e-5, 0.20718812,1.57],
    [ 7e14, 1.15e15, 1.5e15],
    [2e15, 2.44593e16, 3e16]
    ])

logvar = [1,2,4,5]
vars[logvar] = np.log10(vars[logvar])

spec = []
freqs = []
bands = 'LSCXK'
for band in bands:
    fgdir = './fit_26.01.2017/gauss/'
    spec = np.append(spec, np.loadtxt(fgdir + f'spec_{band}.txt'))
    freqs = np.append(freqs, np.loadtxt(fgdir + f'freqs_{band}.txt'))

nu = freqs

#
# The following three arrays MUST be determined and passed to imgpu.Mcgpu.
#
# pdescr1[6]: 1 - nonangular, 2 - angular parameter
#
pdescr1 = np.array((1, 1, 1, 1, 1, 1),  dtype=int32)
# pmint1[6] & pmaxt1[6]: lower and upper limits (the prior boundaries)
pmint1 = np.array(vars[:,0], dtype=float32)
pmaxt1 = np.array(vars[:,2], dtype=float32)


sys.exit(0)

#
# Create the 'rspc' object of the 'Mcgpu' class.
# At creation, all the variables and arrays needed for the MCMC algorithm
# work are created and initialized. They are the rspc 'attributes'
# and they are accessible as rspc.<attributte-name>.
#

rspc = imgpu.Mcgpu(pdescr=pdescr1, pmint=pmint1, pmaxt=pmaxt1)

#
# The burnin_and_search() method does everything. 
#

rspc.burnin_and_search()

rspc.reset_gpu()

#
# The best solution is at the lowest chi2
#

chi2 = rspc.chi2
im = chi2.argmin()   # Indices of the chi^2 minimums 
pout = rspc.pout

print('The minimum chi^2 at chi2[%d] = %f' % (im, chi2[im]))
print('pout[:,%d] = ' % (im), pout[:,im])


#
# Histograms of the all the parameters "random walk" are sometimes
# really helpful. Below is the case of only two parameters, x and y.
#
											  
figure(figsize=(12,6))
subplot(121); hist(pout[0,:], 50, color='b'); grid(1)
xlabel('x')
title(r'MCMC Output Distribution for X')

subplot(122); hist(pout[1,:], 50, color='b'); grid(1)
xlabel('y')
title(r'MCMC Output Distribution for Y')


show()
