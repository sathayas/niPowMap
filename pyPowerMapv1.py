"""
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

    resolutions(self, cmp = False, smoothing = 2):

        calculates volume,euler characteristic,linear diameter, and surface area of the brain image

        input: self (pyPowerMap Object)
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
import nibabel as nib
import skimage
from skimage.measure import *
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class pyPowerMap:

    def __init__(self,data):
        # input: data (str)
        #       - path to mri file of interest in .nii.gz format
        #       example 1: pyPowerMap_Object = pyPowerMap('mymri.nii.gz')
        #       example 2: pyPowerMap_Object = pyPowerMap('C:\Users\Documents\mymri.nii.gz')
        self.data = data
        self.volume = nib.load(self.data).get_data()

    def resolutions(self, cmp = False, smoothing = 2):
        # smooths volume image through gaussian filter
        volume = gaussian(self.volume, smoothing)
        if cmp:
            # Optional:  create subplots comparing interpolated and raw image datas before and after gaussian smoothing
            #   before
            surface = self.volume[:,:,25]
            plt.subplot(1,2,1)
            plt.imshow(surface)
            #   after
            surface_smooth = volume[:,:,25]
            plt.subplot(1,2,2)
            plt.imshow(surface_smooth > .5)
        
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

        return vol,euler,lindiam,surfacearea

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


mri = pyPowerMap('mask.nii.gz')
volume,euler,lindiam,surfacearea = mri.resolutions()
print (volume,euler,lindiam,surfacearea)
mri.visualize()