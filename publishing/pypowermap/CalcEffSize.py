import os
import numpy as np
import math
from .read_vol import read_vol
from .write_vol import write_vol
from .SphereConv import SphereConv
from publishing.pypowermap.reslice import reslice

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

    statHdr, statImg = SphereConv(fOutNaN, fOutSphere, FWHM)



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

    fOut = os.path.join(directory, str(cohenType + "_EffSize_" + file))

    write_vol(statHdr, effSize, fOut)

    return (effSize, fOut)