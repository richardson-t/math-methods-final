import numpy as np
from glob import glob
from astropy.table import Table
from functions import ft_correlation
from tqdm import tqdm

files = glob('sims/logn*')
files.sort()
cfs = np.zeros((100,5))
for f in tqdm(range(len(files))):
    fn,bins = ft_correlation(files[f])
    cfs[f,:] = fn
CF = np.mean(cfs,axis=0)
t = Table()
t.add_column(bins,name='s')
t.add_column(CF,name='CF')
#t.write('avg_FT_CF.fits',overwrite=True)