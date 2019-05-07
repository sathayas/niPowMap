from scipy.ndimage import convolve
import numpy as np
from publishing.pypowermap.write_vol import write_vol

# Performs Spherical Convolution upon Array
# Reference PowerMap/SphereConv.m - https://sourceforge.net/projects/powermap/

def SphereConv(P,Q,r):
    tol = 2.009718347115232*10**-14
    if str(P) == P:
        [hdrP, P] = read_vol(P)
    if hdrP:
        VOX = hdrP['pixdim'][1:4]
        if VOX[0] < 1:
            flipdimFlag = 2
        elif VOX[1] < 1:
            flipdimFlag = 1
        elif VOX[2] < 1:
            flipdimFlag = 3
        else:
            flipdimFlag = 0

        VOX = abs(VOX)
    else:
        VOX = [1,1,1]
        flipdimFlag = 0

    hdrXin = hdrP
    Xin = P

    if flipdimFlag > 0:
        Xin = np.flip(Xin, flipdimFlag)

    bXin = np.ones((np.size(Xin,0),np.size(Xin, 1),np.size(Xin,2)))

    tmp = np.argwhere(np.isnan(Xin))
    tmp2 = np.argwhere(~np.isnan(Xin))
    bXin[bXin == np.nan] = 0
    Xin[Xin == np.nan] = 0

    r2 = r/VOX
    r2[r2 < 1] = 1
    rdim = 2 * r2 + 1
    roff = (r2 + 1)* VOX
    [x,y,z] = np.meshgrid([i for i in range(1,int(round(rdim[0])+1))], [i for i in range(1,int(round(rdim[1])+1))], [i for i in range(1,int(round(rdim[2])+1))])
    x = x * VOX[0]
    y = y * VOX[1]
    z = z * VOX[2]
    rSph = ((x - roff[0])**2 + (y - roff[1])**2 + (z - roff[2])** 2)**(1 / 2)
    brSph = rSph < (r + tol)
    sumSph = sum(brSph)

    Xout = convolve(np.asarray(Xin), brSph)

    sumXout = convolve(bXin, brSph)
    Xout[~np.isnan(Xin)] = Xout[~np.isnan(Xin)]/ sumXout[~np.isnan(Xin)]
    Xout[np.isnan(Xin)] = np.nan
    write_vol(hdrXin, Xout, Q)
    return hdrXin, Xout