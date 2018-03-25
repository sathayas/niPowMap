import numpy as np
from numpy import prod
import os

#I'm not sure what the first line of the Matlab code does, but this is my best guess
effSize, fOut = CalcEffSize(fStat, fMask, statInfo, FWHM, dirOut)

#if the fMask is a numpy array
if (nargin < 2 or fMask.size() == 0):
	mask = 1
else:

	#getting second output from the function
	discard, mask = pm_read_vol(fMask)

lFWHM = np.array(FWHM).size()

if lFWHM >1:
	FWHM = prod(FWHM)**(1/lFWHM)

#directory, file = fileparts(fStat)
with open(fStat) as f:
    for line in f:
        drive, directory = os.path.splitdrive(line)
        directory, filename = os.path.split(path)

file = os.path.splitext(fStat)[0]

#fOutNaN = fullfile(directory,strcat('NaNm_',file)); 

fOutNan = os.path.join(directory, str("NaNm_"+file))

#other function in Matlab
statHdr, statImg = pm_read_vol(fStat)

divided = np.divide(statImg, statImg)

statNaN = np.mulitply(statImg, divided)

#pm_write_vol is another function
pm_write_vol(statHdr, statNaN, fOutNaN)


with open(fOutNan) as f:
    for line in f:
        drive, directory = os.path.splitdrive(line)
        directory, filename = os.path.split(path)

file = os.path.splitext(fStat)[0]

fOutSphere = os.path.join(directory, str("s"+file))

#another previous function
SphereConv(fOutNaN, fOutSphere, FWHM)







