import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

def error(N,A,b):
    return A/np.sqrt(N)+b

covs = glob('cov_*_mocks.fits')
covs.sort()
nmocks = np.linspace(10,100,10).astype(int)
s = np.linspace(0,40,5)

cov_gt = fits.getdata('old_fits/cov_gt.fits')
U,L,V = np.linalg.svd(cov_gt)
cov_gt_sq = U@np.diag(np.sqrt(L))@V
gt = np.linalg.inv(cov_gt_sq)

devs = []

for mock in nmocks:
    matrix = fits.getdata(f'fits/cov_{mock}_mocks.fits')
    T = gt@matrix@gt
    devs.append(np.std(T))

'''
plt.figure()
plt.pcolormesh(s,s,gt@cov_gt@gt)
plt.title('T, mocks=100')
plt.xlabel(r'$S_i$'); plt.ylabel(r'$S_j$')
plt.colorbar()
plt.savefig('T_100.pdf',dpi=300)
plt.close()
'''
popt,pcov = curve_fit(error,nmocks,devs,p0=(2,0.5))
#devs.append(np.std(gt@cov_gt@gt))
plt.plot(nmocks,devs,'o')
plt.plot(nmocks,error(nmocks,*popt),label='1/sqrt fit')
plt.title(r'$C_{\rm gt}^{-1/2}C_{\rm mock}C_{\rm gt}^{-1/2}$')
plt.xlabel(r'$N_{\rm mocks}$')
plt.ylabel('RMS')
plt.legend()
plt.savefig('rms.pdf',dpi=300)