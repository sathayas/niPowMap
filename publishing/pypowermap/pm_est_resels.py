import numpy as np
import nibabel as nib
import sys
import os
from scipy.stats.mstats import gmean
from publishing.pypowermap.pm_est_fwhm import pm_est_fwhm


def est_resels(fwhm_info, mask_file):
    '''
    Function est_resels.py
    Calculates 0-3D resel counts of a binary mask image.

    input parameters:
          fwhm_info:   A 1x3 vector of FWHM in voxels in x, y, and z-directions.
          mask_file:   File name for the binary mask file

    returns:
          resels:      A 1x4 vector of resel counts in 0D, 1D, 2D, and 3D
                       space.
    DETAILS:
          This function is a modified version of mask_resels.m function, originally
          written by Keith Worsley as part of fmristat package. This function has
          been modified to simplify reading / writing of images. Also non-stationary
          smoothness calculation is disabled.
    PROGRAMMER'S NOTE:
          This function calls the following functions, transated from fmristat
          (mainly for resel calculation):
                 mask_mesh.py
                 mesh_tet.py
                 intrinsicvol.py (to be completed)
     Reference PowerMap/pm_est_resels.m - https://sourceforge.net/projects/powermap/
    '''


    # Initialization
    # ----------------------------------------------------------------------------
    mask_thresh = 0
    fwhm_info = np.array(fwhm_info)

    # Rading in the mask image and preparing for resel calculation
    # ----------------------------------------------------------------------------

    # -first, reading in the mask image header info
    mask_img = nib.load(mask_file)

    # removing the extension from the mask file name
    mask_file_path = os.path.abspath(mask_file)
    mask_file_ext = os.path.abspath(mask_file)
    while mask_file_ext != '':
        mask_file_path, mask_file_ext = os.path.splitext(mask_file_path)

    # temporary output file name base
    pth, fname = os.path.split(mask_file_path)
    base = os.path.join(pth, fname + '_coord');

    # -deleting files from previous runs.
    if os.path.exists(base + '.nii.gz'):
        os.unlink(base + '.nii.gz')
    if os.path.exists(base + '_mask.nii.gz'):
        os.unlink(base + '_mask.nii.gz')
    if os.path.exists(base + '_mask.nii.gz'):
        os.unlink(base + '_mask.nii.gz')
    if os.path.exists(base + '_tet.nii.gz'):
        os.unlink(base + '_tet.nii.gz')

    # converting fwhm in mm, and calculating the geometric mean
    dim = mask_img.header['dim'][1:4]
    pixdim = mask_img.header['pixdim'][1:4]
    fwhm = gmean(abs(pixdim) * fwhm_info)

    # creating coordinate images
    mask_data = mask_img.get_data()
    x_ind = range(mask_data.shape[0])
    y_ind = range(mask_data.shape[1])
    z_ind = range(mask_data.shape[2])
    slice_ind = np.meshgrid(x_ind, y_ind)
    vol_ind = np.indices(mask_data.shape)
    #  vol_ind[0]: x-coordinates
    #  vol_ind[1]: y-coordinates
    #  vol_ind[2]: z-coordinates

    # sanity check for the voxel coordinates, manually generated
    # vol_ind = np.zeros(mask_data.shape + (3,));
    # for iz in z_ind:
    #    vol_ind[:,:,iz,0] = slice_ind[0].T
    #    vol_ind[:,:,iz,1] = slice_ind[1].T
    #    vol_ind[:,:,iz,2] = iz

    # converting coordinate images to 1D vectors
    vol_ind_1D = np.vstack([vol_ind[0].flatten(),
                            vol_ind[1].flatten(),
                            vol_ind[2].flatten(),
                            np.ones((1, len(vol_ind[0].flatten())))])

    # voxel coordinates in mm
    vol_mm_1D = np.dot(mask_img.get_sform(), vol_ind_1D)
    vol_mm = np.zeros(mask_data.shape + (3,))
    for iDim in range(3):
        vol_mm[:, :, :, iDim] = vol_mm_1D[iDim].reshape(mask_data.shape)

    # writing out to a 4D image
    fcoord_img = base + '.nii.gz'
    coord_img = nib.Nifti1Image(vol_mm, mask_img.affine)
    nib.save(coord_img, fcoord_img)
