import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi

def reslice(mat,rdims):

    dims = np.shape(mat)

    #dimensions of matrix before reslicing
    nx = dims[1]
    ny = dims[0]
    nz = dims[2]

    # dimensions of resliced matrix
    nmx = rdims[1]
    nmy = rdims[0]
    nmz = rdims[2]

    #create grids defining coordinate systems for mat and rmat
    [x,y,z] = np.meshgrid(np.arange(0,nx)/nx, np.arange(0,ny)/ny, np.arange(0,nz)/nz)
    [xi, yi, zi] = np.meshgrid(np.arange(0, nmx) / nmx, np.arange(0, nmy) / nmy, np.arange(0, nmz) / nmz)

    #interpolate mat(x,y,z) into rmat(xi,yi,zi)
    interp = rgi((x,y,z), mat)
    rmat = interp(np.array([xi,yi,zi]).T)
    rmat = np.nan_to_num(rmat)

    return rmat