import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from astropy.io import fits

covs = glob('cov_*_mocks.fits')
covs.sort()
nmocks = [10,25,50,100]
s = np.linspace(0,40,5)

cov_gt = fits.getdata('cov_gt.fits')
U,L,V = np.linalg.svd(cov_gt)
cov_gt_sq = U@np.diag(np.sqrt(L))@V
gt = np.linalg.inv(cov_gt_sq)

devs = []

for file in range(len(covs)):
    matrix = fits.getdata(covs[file])
    T = gt@matrix@gt
    devs.append(np.std(T))

plt.figure()
plt.pcolormesh(s,s,gt@cov_gt@gt)
plt.title('T, mocks=100')
plt.xlabel(r'$S_i$'); plt.ylabel(r'$S_j$')
plt.colorbar()
plt.savefig('T_100.pdf',dpi=300)
plt.close()

devs.append(np.std(gt@cov_gt@gt))
plt.plot(nmocks,devs,'o')
plt.xlabel(r'$N_{\rm mocks}$')
plt.ylabel('RMS')
plt.savefig('rms.pdf',dpi=300)