import nibabel as nib
import os
import math
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson
from scipy.special import gammaln, gamma, ncfdtr
from scipy.stats import t as T
from scipy.stats import chi2
from scipy.stats import f
from scipy.stats import norm, ncf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import operator

def PowerSurfaceF(fStat, fMask=[], df=[1, 28], FWHM="default", FWEal=.05, dirOut=[]):
    eps = 2.2204 * 10 ** -16
    ximg = nib.load(fStat)
    X = ximg.get_data()
    dir = os.path.split(fStat)[0]
    if FWHM == "default":
        FWHM = est_fwhm(X, df, 'T')

    R = [47,-17.379251775597922,390.9317357004132,1156.787066193768]
    # R = pm_est_resels(FWHM, fMask)
    Rs = [1, 4, 2 * math.pi, (4 / 3) * math.pi]

    numdf = 2
    nc = np.arange(0, 52+ eps, 2)
    Ext_nc = np.arange(60, 210 + eps, 10)
    df = [i for i in range(5, 21)] + [i for i in range(22, 41, 2)] + [i for i in range(45, 101, 5)] + [i for i in
                                                                                                       range(110, 201,
                                                                                                             10)]

    lennc = len(nc)
    lendf = len(df)

    PowSurf = np.zeros((lendf, lennc))
    RecTh = np.zeros((lendf, 1))

    print('Calculating power surface.')
    for idf in range(0, lendf):

        print('.')

        tmpdf = df[idf]
        FWEth = uc_RF(FWEal, [numdf, tmpdf], 'F', R, 1)
        RecTh[idf] = FWEth

        for inc in range(0, lennc):
            tmpnc = nc[inc]
            tmppow = P_ncF(FWEth,numdf,tmpdf,tmpnc,Rs)
            # look back at th.is
            PowSurf[idf, inc] = tmppow[0]

    print('Done!\n')

    for idf in range(0, lendf):

        imaxPow = np.argmax(PowSurf[idf, :])

        if imaxPow + 1 < lennc:
            for inc in range(imaxPow, lennc):
                PowSurf[idf, inc] = PowSurf[idf,imaxPow]

    for inc in range(0, lennc):

        imaxPow = np.argmax(PowSurf[:, inc])

        if imaxPow + 1 < lendf:
            for idf in range(imaxPow + 1, lendf):
                PowSurf[idf, inc] = PowSurf[idf,imaxPow]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(nc, np.asarray(df))
    ax.plot_surface(X, Y, PowSurf)
    ax.invert_xaxis()
    ax.set_zlabel("Power(FWE Corrected)")
    plt.xlabel("Non Centrality")
    plt.ylabel("Degrees of Freedom")
    plt.show()
    return PowSurf

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

def P_RF(c, k, z, df, STAT, R, n):
    eps = 2.2204*10**-16
    D = max(np.nonzero(R)[0])
    R = R[:D+1]

    arange = np.arange(D+1)
    temp = gamma((arange+1) / 2)
    G = np.divide((math.sqrt(math.pi)), temp)
    EC = pm_ECdensity(STAT, z, df)
    EC = EC[:D+1] + eps

    temp2 = np.multiply(EC.T, G)
    temp3 = toeplitz(temp2)

    P = (np.triu(temp3)) ** n
    P = P[0, :]
    EM = np.multiply((np.divide(R, G)), P)
    Em = sum(EM)
    EN = P[0] * R[D]
    En = EN / EM[D]

    D = D - 1

    if (not k or not D):
        p = 1

    elif STAT == 'Z':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    elif STAT == 'T':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    elif STAT == 'X':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    elif STAT == 'F':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    P = 1 - poisson.cdf(c - 1, (Em + eps) * p)

    if (k > 0 and n > 1):
        P = np.array()
        p = np.array()
    if (k > 0 and (STAT == 'X' or STAT == 'F')):
        P = np.array()
        p = np.array()

    return (P, p, Em, En, EN)


def pm_ECdensity(STAT, t, df):
    EC = [0,0,0,0]

    if STAT == 'Z':

        a = 4 * math.log(2)
        b = math.exp(-t**2/2)

        EC[0] = 1 - norm.cdf(t)
        EC[1] = (a ** (1 / 2)) / (2 * math.pi) * b
        EC[2] = a/((2*math.pi)**(3/2))*b*t
        EC[3] = a**(3/2)/((2*math.pi)**2)*b*(t**2 - 1)

    elif STAT == 'T':
        v = df[1]
        a = 4 * math.log(2)
        b = math.exp(gammaln((v + 1) / 2) - gammaln(v / 2))
        c = np.power((1 + np.power(t, 2) / v), ((1 - v) / 2))

        EC[0] = 1 - T.cdf(t,v)
        EC[1] = (a ** (1 / 2)) / (2 * math.pi) * c
        EC[2] = np.multiply(a / (2 * math.pi) ** (3 / 2) * c, t / ((v / 2) ** (1 / 2)) * b)
        EC[3] = np.multiply(a ** (3 / 2) / ((2 * math.pi) ** 2) * c, ((v - 1) * (np.power(t, 2)) / v - 1))

    elif STAT == 'X':
        v = df[1]
        a = (4 * math.log(2)) / (2 * math.pi)
        b = np.multiply(np.power(t, 1 / 2 * (v - 1)), math.exp(-t / 2 - gammaln(v / 2)) / 2 ** ((v - 2) / 2))

        EC[0] = 1 - chi2.cdf(t, v)
        EC[1] = a ** (1 / 2) * b
        EC[2] = np.multiply(a * b, (t - (v - 1)))
        EC[3] = np.multiply(a ** (3 / 2) * b, (np.power(t, 2) - (2 * v - 1) * t + (v - 1) * (v - 2)))

    elif STAT == "F":
        k = df[0]
        v = df[1]
        a = (4 * math.log(2)) / (2 * math.pi)
        b = gammaln(v / 2) + gammaln(k / 2)

        EC[0] = 1 - f.cdf(t, df[0], df[1])
        EC[1] = a**(1/2)*math.exp(gammaln((v+k-1)/2)-b)*2**(1/2)*(k*t/v)**(1/2*(k-1))*(1+k*t/v)**(-1/2*(v+k-2))
        EC[2] = a*math.exp(gammaln((v+k-2)/2)-b)*(k*t/v)**(1/2*(k-2))*(1+k*t/v)**(-1/2*(v+k-2))*((v-1)*k*t/v-(k-1))
        EC[3] = a**(3/2)*math.exp(gammaln((v+k-3)/2)-b)*2**(-1/2)*(k*t/v)**(1/2*(k-3))*(1+k*t/v)**(-1/2*(v+k-2))*((v-1)*(v-2)*(k*t/v)**2-(2*v*k-v-k-1)*(k*t/v)+(k-1)*(k-2))

    return (np.asarray(EC).T)
def pm_u(a,df,STAT):
    if STAT == 'Z':
        return norm.ppf(1-a)

    elif STAT == 'T':
        return T.ppf(1-a, df[1])

    elif STAT == 'X':
        return chi2.ppf(1-a,df[1])

    elif STAT == "F":
        return f.ppf(1-a,df[0],df[1])

    elif STAT == "P":
        return a


def uc_RF(a, df, STAT, R, n):

    u = pm_u((a/sum(R))**(1/n), df,STAT)
    du = 1*10**-6

    d = 1

    while abs(d)>1*10**-6:

        [P,P,p,En,EN] = P_RF(1, 0, u, df, STAT, R, n)
        [P,P,q, En, EN] = P_RF(1, 0, u+du, df, STAT, R, n)
        d = (a-p)/((q-p)/du)
        u = u+d

    d = 1

    while abs(d) > 1*10**-6:
        [P, P, p, En, EN] = P_RF(1, 0, u, df, STAT, R, n)
        [P, P, q, En, EN] = P_RF(1, 0, u + du, df, STAT, R, n)
        d = (a-p)/((q-p)/du)
        u = u+d


    return (u)


def P_ncF(s, df1, df2, delta, R):
    eps = 2.2204 * 10 ** -16
    k = 0
    n = 1
    c = 1

    D = max(np.nonzero(R)[0])
    R = R[:D + 1]

    arange = np.arange(D + 1)
    temp = gamma((arange + 1) / 2)
    G = np.divide((math.sqrt(math.pi)), temp)
    EC = pm_ECncF(s, df1, df2, delta)
    EC = EC[:D + 1] + eps
    # EC = np.ndarray(EC)

    temp2 = np.multiply(EC, G)
    temp3 = toeplitz(temp2)

    P = (np.triu(temp3)) ** n
    P = P[0, :]
    EM = (R / G) * P
    Em = sum(EM)
    EN = P[0] * R[D]
    En = EN / EM[D]

    D = D
    beta = (gamma(D / 2 + 1) / En) ** (2 / D)
    p = math.exp(-beta * (k ** (2 / D)))
    P = 1 - poisson.cdf(c - 1, (Em + eps) * p)

    return [P, Em, En, EN]


def pm_ECncF(s, n1, n2, delta):
    a = 4 * math.log(2)
    b = 2 * math.pi
    c = n1 * s / n2
    d = 1 + c

    tmpEC = np.zeros(4)
    tmpEC[0] = 1 - ncfdtr(s, n1, n2, delta)

    tmpEC[1] = a ** (1 / 2) * b ** (-1 / 2) * 2 * d * c ** (1 / 2) * pm_Exp_ncX(n1 + n2, delta, -1 / 2) * (
                n2 / n1) * ncf.pdf(s, n1, n2, delta)

    Q1 = a * b ** (-1) * 2 * d
    Q2 = pm_Exp_ncX(n1 + n2, delta, -1) * ((n1 - 1) - (n2 - 1) * c + delta)
    Q3 = (n2 / n1) * ncf.pdf(s, n1, n2, delta)
    tmpEC[2] = (-1) * Q1 * Q2 * Q3

    P1 = a ** (3 / 2) * b ** (-3 / 2) * 2 * d * c ** (-1 / 2)
    P2 = pm_Exp_ncX(n1 + n2, delta, -3 / 2)
    P3 = (n1 - 1) * (n1 - 2) - 2 * (n2 - 1) * (n1 - 1 + delta) * c + (n2 - 1) * (n2 - 2) * c ** 2 + delta * (
                2 * n1 - 1 + delta)
    P4 = c * pm_Exp_ncX(n1 + n2, delta, -1 / 2)
    P5 = (n2 / n1) * ncf.pdf(s, n1, n2, delta)
    tmpEC[3] = P1 * (P2 * P3 - P4) * P5

    return tmpEC


def pm_Exp_ncX(df, delta, r):
    eps = 2.2204 * 10 ** -16
    tol = eps ** (7 / 8)

    j = 0
    isum = 0
    iisum = 1
    gamprod = 1
    if r < 0 and abs(r) % 1 < eps:
        bNegInt = 1
    else:
        bNegInt = 0

    while abs(iisum) > tol:
        gam = 1 / gamma(j + 1)
        d = math.exp(gammaln(r + j + (df / 2)) - gammaln(j + (df / 2))) * (delta / 2) ** j

        iisum = gam * d

        isum = isum + iisum

        j = j + 1

    mr = 2 ** r * math.exp(-delta / 2) * isum
    return mr

print(P_ncF(1.056472480462728*10**5,2,5,2,[1,4,6.283185307179586,4.188790204786391]))
ans = PowerSurfaceF('tstat1.nii.gz', 'mask.nii.gz')
print(ans)