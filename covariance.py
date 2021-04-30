import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from astropy.io import fits
from functions import correlation

files = glob('sims/logn*')
files.sort()
nmocks = np.linspace(10,100,10)
nbins = 5

for n in nmocks:
    n = int(n)
    CF = np.zeros((nbins,nbins,n))
    CF_t = np.zeros((nbins,nbins,n))
    for mock in tqdm(range(n)):
        fn,bins = correlation(files[mock])
        CF[:,:,mock],CF_t[:,:,mock] = np.meshgrid(fn,fn)
          
    cov = np.sum(CF*CF_t,axis=-1)/n-np.sum(CF,axis=-1)*np.sum(CF_t,axis=-1)/n**2
    hdu = fits.PrimaryHDU(cov); hdu.writeto(f'fits/cov_{n}_mocks.fits',overwrite=True)

    s = np.linspace(0,40,nbins)
    plt.figure()
    plt.pcolormesh(s,s,cov)
    plt.title('Covariance Matrix, '+r'$N_{\rm mocks}$'+f'={n}')
    plt.xlabel(r'$S_i$'); plt.ylabel(r'$S_j$')
    cb = plt.colorbar()
    plt.savefig(f'covs/cov_{n}.pdf',dpi=300)
    plt.close()

    plt.figure()
    plt.pcolormesh(s,s,np.linalg.inv(cov))
    plt.title('Inverse, '+r'$N_{\rm mocks}$'+f'={n}')
    plt.xlabel(r'$S_i$'); plt.ylabel(r'$S_j$')
    cb = plt.colorbar()
    plt.savefig(f'invs/inv_{n}.pdf',dpi=300)
    plt.close()

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