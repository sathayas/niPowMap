import numpy as np
# from numpy import prod
import os
import math

def CalcEffSize(fStat, fMask, statInfo, FWHM, dirOut):

	#if the fMask is a numpy array can't really check these since functions aren't translated
	
	if (len(locals) < 2 or fMask.size == 0):
		 mask = 1
	else:

		#getting second output from the function
		discard, mask = pm_read_vol(fMask)

	lFWHM = np.array(FWHM).size

	if lFWHM >1:
		FWHM = np.prod(FWHM)**(1/lFWHM)

	directory, file = os.path.split(fStat)
	filename_w_ext = os.path.basename(fStat)
	file, ext = os.path.splitext(filename_w_ext)

	fOutNaN = os.path.join(directory, str("NaNm_"+file))

	#other function in Matlab
	statHdr, statImg = pm_read_vol(fStat)

	divided = np.divide(statImg, statImg)

	statNaN = np.mulitply(statImg, divided)

	pm_write_vol is another function
	pm_write_vol(statHdr, statNaN, fOutNaN)

	directory, file = os.path.split(fOutNaN)

	fOutSphere = os.path.join(directory, str("s"+file))

	#another previous function
	SphereConv(fOutNaN, fOutSphere, FWHM)

	statHdr, statImg = pm_read_vol(fOutSphere)

	if statInfo.type == 'oneT':
		cohenType = 'd'
		effSize = np.divide(statImg, math.sqrt(statInfo.N))

	elif statInfo.type == 'twoT':
		cohenType = 'd'
		effSize = statImg * (math.sqrt(statInfo.N1 + statInfo.N2)/ math.sqrt(statInfo.N1 * statInfo.N2))

	elif statInfo.type == 'reg':
		cohenType = 'd'
		effSize = np.divide(statImg, statInfo.N)

	elif statInfo.type == 'F':
		cohenType = 'f'
		effSize = np.multiply((statInfo.df1/statInfo.df2), statImg)


	if (np.array(mask).size == 1):
		pass
		mask = np.tile(mask, ((effSize).shape, (effSize).shape))

	elif np.array(mask).shape != np.array(effSize).shape:
		pass
		pm_reslice is another function
		mask = pm_reslice(mask, effSize.shape)

	effSize = np.multiply(mask, effSize)

	 fOut is fStat prepended with d_EffSize_ or f_EffSize
	 #this shouldn't work since os.listdir should take a path not a directory
	if (nargin < 5 or (os.listdir(dirOut) == [])):

		# with open(fStat) as f:

	# 		for line in f:
	# 			drive, dirOut = os.path.splitdrive(line)
	# 			dirOut, file = os.path.split(path)

	fOut = os.path.join(directory, str(cohenType+"_EffSize_"+file))

	#pm_write_vol is another function
	pm_write_vol(statHdr, effSize, fOut)

	return(effSize, fOut)
