import numpy as np
from astropy.io import fits
from scipy.special import spherical_jn
import matplotlib.pyplot as plt
from astropy.table import Table

power = np.loadtxt('pk_z0d55_planck15.txt',skiprows=2)
cov = fits.getdata('cov_gt.fits')
cov_inv = np.linalg.inv(cov)
average_CF = Table.read('avg_CF.fits')
FT_CF = Table.read('avg_FT_CF.fits')

k = power[:,0]
p_lin = power[:,1]
p_av = (p_lin[1:]+p_lin[:-1])/2
k_av = (k[1:]+k[:-1])/2
dk = k[1:]-k[:-1]
sigma = 1

s = np.linspace(0,150,151)
integrand = p_av*np.exp(-sigma*k_av**2)
bessel = spherical_jn(0,np.outer(k_av,s))

CF_lin = np.dot(k_av**2*dk*integrand,bessel)/2/np.pi**2
bin_edges = np.linspace(0,40,6)
bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
CF_bin = np.zeros(len(bin_centers))
for i in range(len(bin_centers)):
    lo_ind = np.argmin(abs(s-bin_edges[i]))
    hi_ind = np.argmin(abs(s-bin_edges[i+1]))
    CF_bin[i] = np.mean(CF_lin[lo_ind:hi_ind+1])
#CF_bin = np.interp(bin_centers,s,CF_lin)

num = average_CF['CF']@(cov_inv@CF_bin)
den = CF_bin@(cov_inv@CF_bin)
b_sq = num/den
print(np.sqrt(b_sq))

plt.figure()
plt.plot(bin_centers,CF_bin,label='Model (binned)')
plt.plot(bin_centers,average_CF['CF']/b_sq,label='Measured (Counting)')
plt.plot(bin_centers,FT_CF['CF']/16,label='Measured (FT)')
plt.title(r'$b_{\rm count}$'+f'={str(np.sqrt(b_sq))[:5]}')
plt.xlabel('s (Mpc)'); plt.ylabel('2PCF')
plt.legend()
#plt.savefig('fit_model.pdf',dpi=300)