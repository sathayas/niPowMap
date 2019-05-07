import nibabel as nib
import skimage
from scipy.stats.mstats import gmean
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np

# est_resels_marching_cube(self, cmp = False, smoothing = 2):
#
#             calculates volume,euler characteristic,linear diameter, and surface area of the brain image
#
#             input: self (publishing Object)
#                    fwhm_info (1x3 numpy array): full width half max info
#                    cmp (boolean) - OPTIONAL: if cmp = True, graphs the comparison of raw and gaussian filtered 2d slices of
#                                              the image
#                    smoothing(int) - OPTIONAL: gaussian smoothing index
#
#             output:
#                     euler (int) - euler characteristic
#                     lindiam (int) - linear diameter
#                     surfacearea (int) - surface area
#                     vol (int) - volume

def est_resels_marching_cube(file, fwhm_info, cmp=False, smoothing=2):
    nii = nib.load(file)
    vol = nii.get_data()
    voxelsize = nii.header['pixdim'][1:4]
    binary_volume = (vol != 0).astype(np.float32)
    # smooths volume image through gaussian filter
    volume = gaussian(binary_volume, smoothing)
    if cmp:
        # Optional:  create subplots comparing interpolated and raw image datas before and after gaussian smoothing
        #   before
        surface = binary_volume[:, :, 25]
        plt.subplot(1, 2, 1)
        plt.imshow(surface)
        #   after
        surface_smooth = volume[:, :, 25]
        plt.subplot(1, 2, 2)
        plt.imshow(surface_smooth > .5)

        plt.show()

    # calculates Volume, Euler Characteristic, Linear Diameter
    label = skimage.measure.label(volume > .5)
    rprops = skimage.measure.regionprops(label)
    vol, euler, lindiam = 0, 0, 0

    # sums up Volume, Euler Characteristic, Linear Diameter of all regions
    for i in range(len(rprops)):
        vol += rprops[i].area
        bbox = rprops[i].bbox
        lindiam += (abs(bbox[0] - bbox[3]) + abs(bbox[1] - bbox[4]) + abs(bbox[2] - bbox[5])) / 2
        euler += rprops[i].euler_number

    # calculates surface area using Marching cubes lewiner algorithm
    verts, faces, v, m = skimage.measure.marching_cubes_lewiner(volume, .5)
    surfacearea = skimage.measure.mesh_surface_area(verts, faces)
    fwhm = gmean(fwhm_info * voxelsize)
    lindiam, surfacearea, vol = lindiam / fwhm, surfacearea / (fwhm ** 2), vol / (fwhm ** 3)
    return euler, lindiam, surfacearea, vol