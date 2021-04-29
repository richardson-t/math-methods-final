import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from astropy.io import fits
from functions import correlation, ft_correlation

files = glob('sims/logn*')
files.sort()
nmocks = 50
rmax = 40; ns = 5
si = np.linspace(0,rmax,ns); sj = np.linspace(0,rmax,ns)
cov = np.zeros((ns,ns))

for i in tqdm(range(ns)):
    for j in tqdm(range(i,ns)):
        CF_i = np.zeros(nmocks)
        CF_j = np.zeros(nmocks)
        for index in range(nmocks):
            fn,bins = correlation(files[index])
            CF_i[index] = fn[np.argmin(abs(si[i]-bins))]
            CF_j[index] = fn[np.argmin(abs(sj[j]-bins))]    
        cov[i,j] = np.sum(CF_i*CF_j)/nmocks-np.sum(CF_i)*np.sum(CF_j)/nmocks**2
cov = cov + cov.T - np.diag(np.diag(cov))

hdu = fits.PrimaryHDU(cov); hdu.writeto(f'cov_{nmocks}_mocks.fits')

plt.pcolormesh(si,sj,cov)
plt.title('Covariance Matrix, '+r'$N_{\rm mocks}$'+f'={nmocks}')
plt.xlabel(r'$S_i$'); plt.ylabel(r'$S_j$')
cb = plt.colorbar()
plt.savefig(f'cov_{nmocks}.pdf',dpi=300)

###############################part 1#########################################
#read in data
#points = process_file(files[0])

#plot
'''
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],marker='.',alpha=0.15)
plt.title('Galaxy Positions')
plt.xlabel('x (Mpc)'); plt.ylabel('y (Mpc)'); plt.gca().set_zlabel('z (Mpc)')
plt.savefig('full_plot.pdf',dpi=300,bbox_inches='tight')

#alternatively:
#plt.hist2d(points[0,:],points[1,:],norm=colors.LogNorm(),bins=40)
'''