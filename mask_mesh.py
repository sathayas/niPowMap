import numpy as np
import nibabel as nib
import sys
import os



def mask_mesh(input_file, output_file_base, mask_file):
    '''
    Function mask_mesh.py
    Makes a mesh and mask for input to mesh_tet.py 

    input parameters:
          input_file:           A 4D coordinate image created in est_resels.py.
          output_file_base:     The base for output file names.
          mask_file:            File name for the binary mask file

    returns:
          None



    Originally written for fmristat package by Keith Worsley.
    
    This is a modified version using read/write functions from nibabel.
    There are other modifications to make the program shorter and run faster.

    '''

    # Initialization
    #-------------------------------------------------------------------------------
    mask_thresh   = 0


    #-loading in the xyz coordinate info stored in a 4D file
    #-------------------------------------------------------------------------------
    coord_img = nib.load(input_file)
    dim = coord_img.header['dim']
    n = dim[4]
    numslices = dim[3]
    J = dim[2]
    I = dim[1];

    # removing the extension from the coordinate file name
    coord_file_path = os.path.abspath(input_file)
    coord_file_ext = os.path.abspath(input_file)
    while coord_file_ext!='':
        coord_file_path, coord_file_ext = os.path.splitext(coord_file_path)

    # temporary output file name base
    pth, fname = os.path.split(coord_file_path)
    base = os.path.join(pth,fname);


    #-loading in the mask file
    #-------------------------------------------------------------------------------
    m = nib.load(mask_file)
    

    # Set up:
    #-------------------------------------------------------------------------------
    i = np.kron(np.ones((1,J)),np.arange(I))
    j = np.kron(np.arange(J),np.ones((1,I)))

    IJ = I * J
    ex = np.nonzero(i<I-1)[-1].reshape(1,-1)
    ex1 = np.vstack((ex, ex+IJ)).T
    ex = np.nonzero(i>0)[-1].reshape(1,-1)
    ex2 = np.vstack((ex, ex+IJ)).T

    ey = np.nonzero(j<J-1)[-1].reshape(1,-1)
    ey1 = np.vstack((ey, ey+IJ)).T
    ey = np.nonzero(J>0)[-1].reshape(1,-1)
    ey2 = np.vstack((ey, ey+IJ)).T

    ez = np.arange(IJ)
    ez1 = ez
    ez2 = ez+IJ;

    exye = np.nonzero(((i+j)%2==0) & (i<I-1) & (j<J-1))[-1].reshape(1,-1)
    exyo = np.nonzero(((i+j)%2==1) & (i<I-1) & (j<J-1))[-1].reshape(1,-1)
    exy = np.hstack([exye, exyo])
    exy1 = np.vstack([np.hstack([exye, exyo+1]),
                      np.hstack([exye+1+IJ, exyo+IJ])]).T
    exy2 = np.vstack([np.hstack([exye+1+I, exyo+I]),
                      np.hstack([exye+I+IJ, exyo+1+I+IJ])]).T

    exze = np.nonzero(((i+j)%2==0) & (i<I-1))[-1].reshape(1,-1)
    exzo = np.nonzero(((i+j)%2==1) & (i<I-1))[-1].reshape(1,-1)
    exz = np.hstack([exze, exzo])
    exz1 = np.hstack([exze, exzo+1])
    exz2 = np.hstack([exze+1+IJ, exzo+IJ])



