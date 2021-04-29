import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from astropy.io import fits

covs = glob('cov_*_mocks.fits')
covs.sort()
nmocks = [10,20,50,100]

cov_gt = fits.getdata('cov_gt.fits')
U,L,V = np.linalg.svd(cov_gt)
cov_gt_sq = U@np.diag(np.sqrt(L))@V
gt = np.linalg.inv(cov_gt_sq)

devs = []

for file in covs:
    matrix = fits.getdata(file)
    T = gt@matrix@gt
    devs.append(np.stdev(T))

plt.plot(nmocks,devs)