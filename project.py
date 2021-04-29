import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from glob import glob
from tqdm import tqdm
from functions import correlation, ft_correlation
from mpl_toolkits import mplot3d
from matplotlib import colors

files = glob('sims/logn*')
files.sort()
nmocks = 5

rmax = 40; ns = 5
si = np.linspace(0,rmax,ns); sj = np.linspace(0,rmax,ns)
cov = np.zeros((ns,ns))

for i in tqdm(range(ns)):
    for j in tqdm(range(i,ns)):
        CF_i = np.zeros(nmocks)
        CF_j = np.zeros(nmocks)
        for index in tqdm(range(nmocks)):
            fn,bins = correlation(files[index],rmax)
            CF_i[index] = fn[np.argmin(abs(si[i]-bins))]
            CF_j[index] = fn[np.argmin(abs(sj[j]-bins))]    
        cov[i,j] = np.sum(CF_i*CF_j)/nmocks-np.sum(CF_i)*np.sum(CF_j)/nmocks**2
cov = np.abs(cov + cov.T - np.diag(np.diag(cov)))
plt.pcolormesh(si,sj,cov,norm=colors.LogNorm())
cb = plt.colorbar()
#plt.plot(s,CF)
#plt.title('2PCF')
#plt.xlabel('s'); plt.ylabel(r'$\xi$(s)')
#plt.xscale('log'); plt.yscale('log')

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

#R(s):
#n*4*np.pi/3*s**2
#n = N/V?