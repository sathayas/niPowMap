import numpy as np
from scipy.special import gamma

def pm_est_fwhm(x, df, stat):
    '''
    Function pm_est_fwhm.py
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