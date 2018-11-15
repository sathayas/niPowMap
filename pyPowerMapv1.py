import nibabel as nib
import skimage
from scipy.stats.mstats import gmean
from skimage.measure import *
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import gamma

class pyPowerMap:
    r"""
    pyPowerMap version 0.1

    Developer: Yeji "Charlotte" Yun
    Supervisor: Satoru Hayasaka
    Last Revised: 11/14/2018

    Purpose:

    pyPowermap is  a software tool for calculating statistical power and sample sizes for neuroimaging studies. This is a
    Python implementation of the original PowerMap project written in MATLAB. Our goal is to provide access to the PowerMap
    functionality to non-MatLab users at improved efficiency.

    Methods:
        __init__(self, data): constructor
            input: data (str)
                  - path to mri file of interest in .nii.gz format
                  example 1: pyPowerMap_Object = pyPowerMap('mymri.nii.gz')
                  example 2: pyPowerMap_Object = pyPowerMap('C:\Users\Documents\mymri.nii.gz')

        est_fwhm(self, df, stat):

        Calculates smoothness of a statistic image (T- or F-statistic) directly
        from a statistic image rather than from residuals.
        input:
              x (nparray):        A 3D array of statistic image data.
              df (List[int]):       A 1x2 vector of degrees of freedom, with [df1 df2]. For an
                        F-image, df1 and df2 correspond to the numerator and demoninator
                        degrees of freedom, respectively. For a T-image, df1=1 and
                        df2 is the error df.
              stat (str):     Statistical image type
                        'T' - T-image
                        'F' - F-image
        output:
              fwhm (nparray):     A 1x3 vector of FWHM in terms of voxels, in x, y, and z
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

        resels(self, cmp = False, smoothing = 2):

            calculates volume,euler characteristic,linear diameter, and surface area of the brain image

            input: self (pyPowerMap Object)
                   fwhm_info (1x3 numpy array): full width half max info
                   cmp (boolean) - OPTIONAL: if cmp = True, graphs the comparison of raw and gaussian filtered 2d slices of
                                             the image
                   smoothing(int) - OPTIONAL: gaussian smoothing index

            output: vol (int) - volume
                    euler (int) - euler characteristic
                    lindiam (int) - linear diameter
                    surfacearea (int) - surface area

        void visualize(self): graphs 3d plot of the image
            input: self (pyPowerMap Object)
    """
    def __init__(self,data):
        # input: data (str)
        #       - path to mri file of interest in .nii.gz format
        #       example 1: pyPowerMap_Object = pyPowerMap('mymri.nii.gz')
        #       example 2: pyPowerMap_Object = pyPowerMap('C:\Users\Documents\mymri.nii.gz')
        self.data = data
        self.nii = nib.load(self.data)
        self.volume = self.nii.get_data()
        self.voxel_size = self.nii.header['pixdim'][1:4]

    def est_fwhm(self, df, stat):

        # First, checking the input
        x= self.volume
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

    def resels(self, fwhm_info, cmp = False, smoothing = 2):
        binary_volume = (self.volume != 0).astype(np.float32)
        # smooths volume image through gaussian filter
        volume = gaussian(binary_volume, smoothing)
        if cmp:
            # Optional:  create subplots comparing interpolated and raw image datas before and after gaussian smoothing
            #   before
            surface = binary_volume[:,:,25]
            plt.subplot(1,2,1)
            plt.imshow(surface)
            #   after
            surface_smooth = volume[:,:,25]
            plt.subplot(1,2,2)
            plt.imshow(surface_smooth > .5)

            plt.show()
        
        # calculates Volume, Euler Characteristic, Linear Diameter
        label = skimage.measure.label(volume>.5)
        rprops = skimage.measure.regionprops(label)
        vol,euler,lindiam = 0,0,0

        # sums up Volume, Euler Characteristic, Linear Diameter of all regions
        for i in range(len(rprops)):
            vol += rprops[i].area
            bbox = rprops[i].bbox
            lindiam += (abs(bbox[0] - bbox[3]) + abs(bbox[1] - bbox[4]) + abs(bbox[2] - bbox[5])) / 2
            euler += rprops[i].euler_number

        # calculates surface area using Marching cubes lewiner algorithm
        verts, faces, v, m = skimage.measure.marching_cubes_lewiner(volume,.5)
        surfacearea = skimage.measure.mesh_surface_area(verts,faces)
        fwhm = gmean(fwhm_info*self.voxel_size)
        lindiam,surfacearea,vol = lindiam/fwhm, surfacearea/(fwhm**2), vol/(fwhm**3)
        print("Euler Characteristic = " + str(euler))
        print("Linear Diameter = " + str(lindiam))
        print("Surface Area = " + str(surfacearea))
        print("Volume = " + str(vol))
        return euler,lindiam,surfacearea,vol

    def visualize(self):
        # graphs 3d plot of the brain image
        verts, faces, v, m = skimage.measure.marching_cubes_lewiner(self.volume,.5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.set_xlim(0,80) # a = 6 (times two for 2nd ellipsoid)
        ax.set_ylim(50,150)  # b = 10
        ax.set_zlim(0,60)
        plt.show()


#mri = pyPowerMap('mask.nii.gz')
#euler,lindiam,surfacearea,volume = mri.resels(np.asarray([1,1,1]))
#print (volume,euler,lindiam,surfacearea)
#mri.visualize()
mri2 = pyPowerMap('tstat1.nii.gz')
fwhm = mri2.est_fwhm([1,10],'F')
print(fwhm)
euler,lindiam,surfacearea, volume= mri2.resels(fwhm)
#mri2.visualize()
