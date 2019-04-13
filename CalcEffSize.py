import os
import numpy as np
import math
import nibabel as nib
from scipy.special import gamma
from scipy.ndimage import convolve

def CalcEffSize(fStat, fMask, statInfo, FWHM, dirOut):
    nargin = len(locals())

    if (nargin < 2 or fMask == ""):
        mask = 1
    else:
        # getting second output from the function
        discard, mask = read_vol(fMask)

    lFWHM = np.array(FWHM).size

    if lFWHM > 1:
        FWHM = np.prod(FWHM) ** (1 / lFWHM)

    directory, file = os.path.split(fStat)

    filename_w_ext = os.path.basename(fStat)
    file, ext = os.path.splitext(filename_w_ext)

    fOutNaN = os.path.join(directory, str(file))

    statHdr, statImg = read_vol(fStat)

    statNaN = statImg.astype('float')
    statNaN[statNaN == 0] = np.nan

    write_vol(statHdr, statNaN, fOutNaN)

    directory, file = os.path.split(fOutNaN)
    filename_w_ext = os.path.basename(fOutNaN)
    file, ext = os.path.splitext(filename_w_ext)

    fOutSphere = os.path.join(directory, str(file))
    fOutSphere2 = os.path.join(directory, str(filename_w_ext))

    SphereConv(fOutNaN, fOutSphere, FWHM)

    statHdr, statImg = read_vol(fOutSphere2)

    if statInfo["type"] == 'oneT':
        cohenType = 'd'
        effSize = np.divide(statImg, math.sqrt(statInfo["N"]))

    elif statInfo["type"] == 'twoT':
        cohenType = 'd'
        effSize = statImg * (math.sqrt(statInfo["N1"] + statInfo["N2"]) / math.sqrt(statInfo["N1"] * statInfo["N2"]))

    elif statInfo["type"] == 'reg':
        cohenType = 'd'
        effSize = np.divide(statImg, statInfo["N"])

    elif statInfo["type"] == 'F':
        cohenType = 'f'
        effSize = np.multiply((statInfo["df1"] / statInfo["df2"]), statImg)

    if (np.array(mask).size == 1):
        mask = np.tile(mask, ((effSize).shape, (effSize).shape))

    elif np.array(mask).shape != np.array(effSize).shape:

        # pm_reslice is another function
        mask = reslice(mask, effSize.shape)

    effSize = np.multiply(mask, effSize)
    # effSize[21, 21, 32] is 0 while it is .1360 in Matlab

    # fOut is fStat prepended with d_EffSize_ or f_EffSize
    # this shouldn't work since os.listdir should take a path not a directory
    cwd = os.getcwd()

    # no documentation of what dirOut is supposed to be.
    # if it is a path, replace cwd with dirOut
    # If it is a string I have no replacement for it, so this is a work-around
    if (nargin < 6 or (len(os.listdir(cwd)) == 0)):
        directory, file = os.path.split(fStat)

        filename_w_ext = os.path.basename(fStat)
        file, ext = os.path.splitext(filename_w_ext)

    fOut = os.path.join(directory, str('a'+cohenType + "_EffSize_" + file))

    write_vol(statHdr, effSize, fOut+'2')

    return (effSize, fOut)

def read_vol(fName):
	# example_filename = os.path.join(data_path, fName)
	#directory, file = os.path.split(fName)
	#filename_w_ext = os.path.basename(fName)
	#file, ext = os.path.splitext(filename_w_ext)
	#example_filename = os.path.join(directory, filename_w_ext)

	img = nib.load(fName)
	data = img.get_data()
	header = img.header

	return(header, data)

def write_vol(V, X, fName):

	X = np.array(X)
	data = X
	img1 = nib.Nifti1Image(data, np.eye(4))
	img1.header['scl_slope'] = 1
	img1.header['scl_inter'] = 0

	directory, file = os.path.split(fName)
	filename_w_ext = os.path.basename(fName)
	file, ext = os.path.splitext(filename_w_ext)
	if filename_w_ext[-4:] == '.nii':
		nib.save(img1, os.path.join(directory, filename_w_ext))
	else:
		filename_w_ext+=".nii"
	nib.save(img1, os.path.join(directory, filename_w_ext))

	return

def reslice(mat,rdims):
    #incomplete
    return

def est_fwhm(x, df, stat):
    '''
    Function est_fwhm.py
    Calculates smoothness of a statistic image (T- or F-statistic) directly
    from a statistic image rather than from residuals.
    input parameters:
          x:        A 3D array of statistic image data.
          df:       A 1x2 vector of degrees of freedom, with [df1 df2]. For an
                    F-image, df1 and df2 correspond to the numerator and demoninator
                    degrees of freedom, respectively. For a T-image, df1=1 and
                    df2 is the error df.
          stat:     Statistical image type
                    'T' - T-image
                    'F' - F-image
    returns:
          fwhm:     A 1x3 vector of FWHM in terms of voxels, in x, y, and z
                    directions.
    DETAILS:
          The fwhm value is derived from the roughness matrix Lambda. Lambda is the
          correlation matrix of gradient of a Gaussian field in x, y, and z
          directions, and in a typical SPM analysis Lambda is derived from
          residual images. In this function, Lambda is derived from a statistic
          image directly using the theoretical expression of the grandient of a
          T-field and an F-field in [1]. This can be done by scaling the covariance
          matrix of numerical grandients of a statistic image appropriately. Based
          on Lambda, fwhm is calculated and returned as an output of this function.
    REFERENCE:
          [1]. Worsley KJ.
                  Local maxima and the expected Euler characteristic of excursion sets
                  of chi-square, F and t fields.
                  Advances in Applied Probability, 26: 13-42 (1994)
    '''


    # First, checking the input
    if stat == 'T' and np.isscalar(df):
        df1 = 1
        df2 = df
    else:
        df1, df2 = df

    dfsum = df1 + df2

    if (df2 <= 4) or ((stat == 'F') and (df2 <= 6)):
        print('Degrees of freedom is too small!')
        return

    x = np.array(x)  # making sure that the input is an array

    # defining necessary parameters for calculation
    # ------------------------------------------------------------------------------------

    tol = 0.000000000001  # a tiny tiny value

    xdim, ydim, zdim = x.shape

    # estimating moments necessary for smoothness estimation
    # done by braking up the theoretical moments of T- or F-field into three parts.
    # ------------------------------------------------------------------------------------

    # -first part X
    if stat == 'F':
        muX = (df1 + df2 - 2) * (gamma((df1 + 1) / 2) * gamma((df2 - 3) / 2)) / (gamma(df1 / 2) * gamma(df2 / 2))
        varX = (df1 + df2 - 2) * (df1 + df2 - 4) * (df1 / 2) / (
                (df2 / 2 - 1) * (df2 / 2 - 2) * (df2 / 2 - 3)) - muX ** 2
    elif stat == 'T':
        muX = df2 ** (1 / 2) * (df2 - 1) / (df2 - 2)
        varX = 2 * df2 * (df2 - 1) / ((df2 - 2) ** 2 * (df2 - 4))
    else:
        print('Unknown statistical field!')
        return

    # -second part Y
    muY = 2 ** (-1 / 2) * gamma((dfsum - 1) / 2) / gamma(dfsum / 2);
    varY = 1 / (dfsum - 2) - muY ** 2;

    # -scaling factor for var(derivative) matrix
    Dscale = 1 / (varX * varY + muX ** 2 * varY + muY ** 2 * varX + muX ** 2 * muY ** 2);
    if stat == 'F':
        Dscale = (df1 / df2) ** 2 * Dscale

    # -Smoothness estimation
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # -NaN masking the image
    x[x == 0] = np.nan

    # -allocating spaced for Lambda calculation
    dx = np.zeros([xdim, ydim, zdim])  # -initializing deriv in x direction
    dy = np.zeros([xdim, ydim, zdim])  # -initializing deriv in y direction
    dz = np.zeros([xdim, ydim, zdim])  # -initializing deriv in z direction

    # -Deriv in x direction
    dx[:(xdim - 1), :, :] = np.diff(x, axis=0)  # -Deriv in x direction
    dx[xdim - 1, :, :] = dx[xdim - 2, :, :]  # -Imputing the edge
    meandx = np.sum(dx[np.isfinite(dx)]) / len(dx[np.isfinite(dx)])
    QzeroX = np.where(np.isnan(dx))
    dx[QzeroX] = 0  # -zeroing NaNs

    # -Deriv in y direction
    dy[:, :(ydim - 1), :] = np.diff(x, axis=1)  # -Deriv in y direction
    dy[:, ydim - 1, :] = dy[:, ydim - 2, :]  # -Imputing the edge
    meandy = np.sum(dy[np.isfinite(dy)]) / len(dy[np.isfinite(dy)])
    QzeroY = np.where(np.isnan(dy))
    dy[QzeroY] = 0  # -zeroing NaNs

    # -Deriv in z direction
    dz[:, :, :(zdim - 1)] = np.diff(x, axis=2)  # -Deriv in z direction
    dz[:, :, zdim - 1] = dy[:, :, zdim - 2]  # -Imputing the edge
    meandz = np.sum(dz[np.isfinite(dz)]) / len(dz[np.isfinite(dz)])
    QzeroZ = np.where(np.isnan(dz))
    dz[QzeroZ] = 0  # -zeroing NaNs

    # -elements of var(derivative) matrix
    Dxx = np.sum((dx[np.nonzero(dx)] - meandx) ** 2) / len(dx[np.nonzero(dx)])
    Dyy = np.sum((dy[np.nonzero(dy)] - meandy) ** 2) / len(dy[np.nonzero(dy)])
    Dzz = np.sum((dz[np.nonzero(dz)] - meandz) ** 2) / len(dz[np.nonzero(dz)])
    Qxy = np.nonzero(dx * dy)
    Qxz = np.nonzero(dx * dz)
    Qyz = np.nonzero(dy * dz)
    Dxy = sum((dx[Qxy] - meandx) * (dy[Qxy] - meandy)) / len(Qxy[0])
    Dxz = sum((dx[Qxz] - meandx) * (dz[Qxz] - meandz)) / len(Qxz[0])
    Dyz = sum((dy[Qyz] - meandy) * (dz[Qyz] - meandz)) / len(Qyz[0])

    D = np.array([[Dxx, Dxy, Dxz], [Dxy, Dyy, Dyz], [Dxz, Dyz, Dzz]])

    # -finally Lambda matrix
    Lambda = Dscale * D

    # -calculating global FWHM
    fwhm = (4 * np.log(2)) ** (3 / 2) / np.diagonal(Lambda) ** (1 / 6)

    return fwhm

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
    write_vol(hdrXin, Xout, Q+'3')

ximg = nib.load('tstat1.nii.gz')
X = ximg.get_data()
FWHM = est_fwhm(X, [1,28], 'T')
statinfo = {"type":"oneT", "N":29, "N1":1, "N2": 1, "df1":1, "df2":28}
print(CalcEffSize('tstat1.nii.gz', 'mask.nii.gz', statinfo, FWHM,""))