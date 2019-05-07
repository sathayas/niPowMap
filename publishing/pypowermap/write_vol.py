import nibabel as nib
import numpy as np
import os
# -----------------------------------------------------------------------
# 
# WRITE_VOL
#
# Purpose   Wrapper function to write a 3D or 4D analyze or nifti image
#           using nibabel
#
# Inputs    V       - Image header
#           X       - Image data
#           fName   - Output file name, including directory
# -----------------------------------------------------------------------
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