import numpy as np
import nibabel as nib
import sys
import os
from est_fwhm import est_fwhm

mask_file = 'mask.nii.gz'
# ftstat = 'tstat1.nii.gz'
# df = 6
# stat = 'T'

ftstat = 'FA_Tstat_YoungOld_DF28.nii.gz'
df = 28
stat = 'T'

tstat_img = nib.load(ftstat)
tstat_data = tstat_img.get_data()
fwhm_info = est_fwhm(tstat_data, df, stat)

print(fwhm_info)

# input_file = 'mask_coord.nii.gz'
