addpath ../MatlabCodes;

mask_file = 'mask.nii.gz';
ftstat = 'tstat1.nii.gz';
df = 6;
stat = 'T';
[V,X] = pm_read_vol(ftstat);

fwhm_info = pm_est_fwhm(X, df, stat);

input_file = 'mask_coord.nii.gz'



