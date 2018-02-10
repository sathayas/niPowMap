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
    I = dim[1]

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

    eyze = np.nonzero(((i+j)%2==0) & (j<J-1))[-1].reshape(1,-1)
    eyzo = np.nonzero(((i+j)%2==0) & (j<J-1))[-1].reshape(1,-1)
    eyz = np.hstack([eyze, eyzo])
    eyz1 = np.hstack([eyze, eyzo + I])
    eyze2 = np.hstack([eyze+I+IJ, eyzo+IJ])

    edges_start1 = np.hstack([(ex1[:,0]).T, (ey1[:,0]).T, (exy1[:,0]).T, (ex2[:,0]).T, (ey2[:,0]).T, (exy2[:,0]).T])
    edges_start2 = np.hstack([(ex2[:,0]).T, (ey2[:,0]).T, (exy2[:,0]).T, (ex1[:,0]).T, (ey1[:,0]).T, (exy1[:,0]).T])

    edge1 = np.hstack([(ex1[:,0]).T, (ey1[:,0]).T, (exy1[:,0]).T, ez1, exz1, eyz1, (ex1[:,1]).T, (ey1[:,1]).T, (exy1[:,1]).T])
    edge2 = np.hstack([(ex2[:,0]).T, (ey2[:,0]).T, (exy2[:,0]).T, ez2, exz2, eyz2, (ex2[:,1]).T, (ey2[:,1]).T, (exy2[:,1]).T])

    edges1 = np.hstack([edge1, edge2])
    edges2 = np.hstack([edge2, edge1])

    #START:
    u = np.zeros(2*IJ,n)
    v = np.zeros(2*IJ,n)
    mask = np.zeros(2*IJ, 1)
    nask = np.zeros(2*IJ, 1)
