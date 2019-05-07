import math
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from publishing.pypowermap.pm_est_fwhm import pm_est_fwhm
from publishing.pypowermap.pm_P_ncT import pm_P_ncT
from publishing.pypowermap.uc_RF import uc_RF

# -----------------------------------------------------------------------
# PowerSurfaceT
# 
#   Purpose     Creates a power surface for a t-statistic image.
#
#   Inputs      - fStat     T-statistic image
#               - fMask     Mask image
#               - FWHM      Full-width at half-max in voxels. If omitted, 
#                           will be calculated by pm_est_fwhm. 
#               - FWEal     Threshold (default is 0.05). fwh
#               - dirOut    Output directory. 
#
#   Outputs     A surface image is created, and a mat file is saved to the
#               statistic image directory.
# -----------------------------------------------------------------------
# Reference PowerMap/PowerSurfaceT.m - https://sourceforge.net/projects/powermap/

def PowerSurfaceT(fStat, fMask=[], df=[1, 28], FWHM="default", FWEal=.05, dirOut=[]):
    eps = 2.2204 * 10 ** -16
    ximg = nib.load(fStat)
    X = ximg.get_data()
    dir = os.path.split(fStat)[0]
    if FWHM == "default":
        FWHM = pm_est_fwhm(X, df, 'T')

    R = [47, -109.416922663253, 15495.5776301671, 288678.000000000]
    # R = pm_est_resels(FWHM, fMask)
    Rs = [1, 4, 2 * math.pi, (4 / 3) * math.pi]

    nc = np.arange(0, 8.2 + eps, 0.2)
    Ext_nc = np.arange(8.5, 20.5 + eps, 0.5)
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
        FWEth = uc_RF(FWEal, [1, tmpdf], 'T', R, 1)
        RecTh[idf] = FWEth

        for inc in range(0, lennc):
            tmpnc = nc[inc]
            tmppow = pm_P_ncT(FWEth, tmpdf, tmpnc, Rs)
            # look back at this
            PowSurf[idf, inc] = tmppow[0]

    print('Done!\n')

    for idf in range(0, lendf):

        imaxPow = np.argmax(PowSurf[idf, :])

        dPow = PowSurf[idf, imaxPow] - PowSurf[idf, imaxPow - 1]
        dPow2 = PowSurf[idf, imaxPow - 1] - PowSurf[idf, imaxPow - 2]
        ddPow = dPow - dPow2

        if imaxPow + 1 < lennc:
            for inc in range(imaxPow, lennc):
                tmpdPow = PowSurf[idf, inc - 1] - PowSurf[idf, inc - 2]
                tmpdPow2 = tmpdPow + ddPow
                if tmpdPow2 > 0:
                    PowSurf[idf, inc] = PowSurf[idf, inc - 1] + tmpdPow2
                else:
                    PowSurf[idf, inc] = PowSurf[idf, inc - 1]

    for inc in range(0, lennc):

        imaxPow = np.argmax(PowSurf[:, inc])

        dPow = PowSurf[imaxPow, inc] - PowSurf[imaxPow - 1, inc]
        dPow2 = PowSurf[imaxPow - 1, inc] - PowSurf[imaxPow - 2, inc]
        ddPow = dPow - dPow2

        if imaxPow + 1 < lendf:
            for idf in range(imaxPow + 1, lendf):
                tmpdPow = PowSurf[idf - 1, inc] - PowSurf[idf - 2, inc]
                tmpdPow2 = tmpdPow + ddPow
                if tmpdPow2 > 0:
                    PowSurf[idf, inc] = PowSurf[idf - 1, inc] + tmpdPow2
                else:
                    PowSurf[idf, inc] = PowSurf[idf - 1, inc]

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