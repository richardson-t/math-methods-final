import numpy as np
from astropy.io import fits
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator,LinearNDInterpolator
from scipy.fft import fftn,ifftn

def process_file(filename):
    data = fits.getdata(filename)
    x = [point[0][0] for point in data]
    y = [point[0][1] for point in data]
    z = [point[0][2] for point in data]
    return np.array((x,y,z)).T

def shiftbox(data,updown,rightleft,inout,R):
    return data[:,0]+updown*R, data[:,1]+rightleft*R, data[:,2]+inout*R

def secondaries(data,V):
    shifts = [-1,0,1]
    x = []; y = []; z = []
    for k in shifts:
        for j in shifts:
            for i in shifts:
                new_x, new_y, new_z = shiftbox(data,i,j,k,int(V**(1/3)))
                x.extend(new_x); y.extend(new_y); z.extend(new_z)
    sec_array = np.array((x,y,z)).T
    sec_tree = KDTree(sec_array)
    return sec_tree, sec_array

def correlation(file,r_max=40,nbins=5,V=1e6):
    data = process_file(file)
    r_max = 40
    sec_tree, sec_array = secondaries(data,V)
    pairs = sec_tree.query_ball_point(data,r_max)
    
    bin_counts = np.zeros(nbins)
    bin_edges = np.linspace(0,r_max,nbins+1)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
    npoints,nds = data.shape
    for index in range(npoints):
        point_list = sec_array[pairs[index]]
        diff = point_list-data[index]
        s = np.sqrt(np.sum(diff**2,axis=1))
        s.sort()
        bin_counts += np.histogram(s[1:],bins=bin_edges)[0]
    
    n = npoints/V
    DCF = bin_counts/npoints
    RCF = n*(bin_edges[1:]**3-bin_edges[:-1]**3)
    CF = DCF/RCF-1
    
    return CF,bin_centers

def counts(data,grid_points):
    data = data-np.array([50,50,50]) #center data in box
    npoints,nds = data.shape
    ngrid = len(grid_points)
    D = np.zeros((ngrid,ngrid,ngrid))
    for point in range(npoints):
        xloc = np.argmin(abs(grid_points-data[point,0]))
        yloc = np.argmin(abs(grid_points-data[point,1]))
        zloc = np.argmin(abs(grid_points-data[point,2]))
        D[xloc,yloc,zloc] += 1
    return D

def counts_interp(data,X,Y,Z):
    npoints,nds = data.shape
    data = data-np.array([50,50,50]) #center data in box
    interp = NearestNDInterpolator(data,np.ones(npoints))
    counts = interp(X,Y,Z)
    return counts

def bin_func(grid_points,r_in,r_out):
    X,Y,Z = np.meshgrid(grid_points,grid_points,grid_points)
    R = np.sqrt(X**2+Y**2+Z**2)
    phi = np.zeros(R.shape)
    norm = 4*np.pi/3*(r_out**3-r_in**3)
    inBin = (R > r_in) & (R < r_out)
    phi[inBin] = 1/norm
    return phi

def ft_correlation(file,r_max=40,nbins=5,V=1e6):
    data = process_file(file)
    
    bin_edges = np.linspace(0,r_max,nbins+1)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
    DCF = np.zeros(nbins)
    
    grid_points = np.linspace(-50,50,101)
    pad = 300; init = len(grid_points)
    D = counts(data,grid_points)
    D_tr = fftn(D,s=[pad,pad,pad])
    
    for i in range(nbins):
        phi = bin_func(grid_points,bin_edges[i],bin_edges[i+1])
        product = D_tr*np.conjugate(fftn(phi,s=[pad,pad,pad]))
        integrand = D*np.real(ifftn(product,s=[init,init,init]))
        DCF[i] = np.sum(integrand)
            
    npoints,nds = data.shape
    n = npoints/V
    RCF = n*(bin_edges[1:]**3-bin_edges[:-1]**3)
    CF = DCF/RCF-1
    
    return CF,bin_centers