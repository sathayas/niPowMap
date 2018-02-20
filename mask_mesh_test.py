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
    img_data = coord_img.get_data()
    dim = coord_img.header['dim']
    n = dim[4]
    numslices = dim[3]
    J = dim[2]
    I = dim[1]

    m = nib.load(mask_file)
    m_img = m.get_data()
    mm_img = m_img.copy()


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
    # m = nib.load(mask_file)
    # m_data = m.get_data()
    m = nib.load(mask_file)
    m_img = m.get_data()
    mm_img = m_img.copy()

    dd = nib.load(input_file)

    dim2 = dd.header['dim']
    dim2[0:4] = np.hstack([I, J, numslices, n])
    dd_img = np.zeros(dim2[0:4])

    

    # Set up:
    #-------------------------------------------------------------------------------
    i = np.kron(np.ones((1,J)),np.arange(I)+1)
    j = np.kron(np.arange(J)+1,np.ones((1,I)))

    IJ = I * J
    ex = np.nonzero(i<I)[-1].reshape(1,-1)
    ex = ex+1
    ex1 = np.vstack((ex, ex+IJ)).T
  

    ex = np.nonzero(i>1)[-1].reshape(1,-1)
    ex = ex+1
    ex2 = np.vstack((ex, ex+IJ)).T


    ey = np.nonzero(j<J)[-1].reshape(1,-1)
    ey = ey+1 
    ey1 = np.vstack((ey, ey+IJ)).T

    ey = np.nonzero(j>1)[-1].reshape(1,-1)
    ey = ey+1

    ey2 = np.vstack((ey, ey+IJ)).T
    # print(ey2.shape) 9828x2

    ez = np.arange(IJ).reshape(1,-1) #added reshape
    # print(ez.shape) 1x9919
    ez = ez+1
    
    ez1 = ez.copy()
    ez2 = ez+IJ

    exye = np.nonzero(((i+j)%2==0) & (i<I) & (j<J))[-1].reshape(1,-1)
    exye = exye+1
    # print(exye.shape) 1x4860

    exyo = np.nonzero(((i+j)%2==1) & (i<I) & (j<J))[-1].reshape(1,-1)
    # print(exyo.shape) 1x4860
    exyo = exyo+1
    exy = np.hstack([exye, exyo])
    # print(exy.shape) 1x9720
 
    exy1 = np.vstack([np.hstack([exye, exyo+1]),
                      np.hstack([exye+1+IJ, exyo+IJ])]).T
    # print(exy1.shape) 9720x2


    exy2 = np.vstack([np.hstack([exye+1+I, exyo+I]),
                      np.hstack([exye+I+IJ, exyo+1+I+IJ])]).T


    exze = np.nonzero(((i+j)%2==0) & (i<I))[-1].reshape(1,-1)
    # print(exze.shape) 1x4905
    exze = exze+1
    exzo = np.nonzero(((i+j)%2==1) & (i<I))[-1].reshape(1,-1)
    # print(exzo.shape) 1x4905
    exzo = exzo +1
    exz = np.hstack([exze, exzo])

    exz1 = np.hstack([exze, exzo+1])
    exz2 = np.hstack([exze+1+IJ, exzo+IJ])

    eyze = np.nonzero(((i+j)%2==0) & (j<J))[-1].reshape(1,-1)
    # print(eyze.shape) 1x4914
    eyze = eyze +1
    eyzo = np.nonzero(((i+j)%2==1) & (j<J))[-1].reshape(1,-1)
    # print(eyzo.shape) 1x4914
    eyzo = eyzo +1
    eyz = np.hstack([eyze, eyzo])
    # print(eyz.shape) 1x9828
    eyz1 = np.hstack([eyze, eyzo + I])
    # print(eyz1.shape) 1x9828
    eyz2 = np.hstack([eyze+I+IJ, eyzo+IJ])

    edges_start1 = np.hstack([(ex1[:,0]).T, (ey1[:,0]).T, (exy1[:,0]).T, (ex2[:,0]).T, (ey2[:,0]).T, (exy2[:,0]).T]).reshape(1,-1)
    edges_start2 = np.hstack([(ex2[:,0]).T, (ey2[:,0]).T, (exy2[:,0]).T, (ex1[:,0]).T, (ey1[:,0]).T, (exy1[:,0]).T]).reshape(1, -1)

    edge1 = np.hstack([(ex1[:,0]).T.reshape(1, -1), (ey1[:,0]).T.reshape(1, -1), (exy1[:,0]).T.reshape(1, -1), ez1, exz1, eyz1, (ex1[:,1]).T.reshape(1,-1), (ey1[:,1]).T.reshape(1,-1), (exy1[:,1]).T.reshape(1, -1)]) #added reshape
    edge2 = np.hstack([(ex2[:,0]).T.reshape(1,-1), (ey2[:,0]).T.reshape(1, -1), (exy2[:,0]).T.reshape(1,-1), ez2, exz2, eyz2, (ex2[:,1]).T.reshape(1, -1), (ey2[:,1]).T.reshape(1,-1), (exy2[:,1]).T.reshape(1, -1)]) #added reshape

    # print(edge1.shape) 1x88273


    edges1 = np.hstack([edge1, edge2])
    edges1 = edges1.reshape(1, -1)

    edges2 = np.hstack([edge2, edge1])

    #START:

    u = np.zeros((2*IJ,n))
    v = np.zeros((2*IJ,n))
    mask = np.zeros((2*IJ, 1))
    nask = np.zeros((2*IJ, 1))
    flip = 1
    print('Calculating mask and mesh')

    for slice in range(2):
    # for slice in range (int(numslices)):
        print('.', end = '')
        flip = 3-flip
        tmpimg = np.array(img_data[:,:,slice,:n])
        tmpimg = tmpimg.reshape(I, J, 1, n)
        # print(tmpimg.shape) 91x109x1x3



        alteration = (flip -1)*IJ
        u[(np.arange(IJ)) + alteration, :] = tmpimg.reshape( IJ, n)


        # print((u[np.arange(IJ)+ alteration, :])[0:5,0:3])
        v[np.arange(IJ)+ alteration, :] = tmpimg.reshape( IJ, n)

        mask[np.arange(IJ)+ alteration, :] = m_img[:, :, slice].reshape(IJ, 1)
        nask[np.arange(IJ)+ alteration, :] = (m_img[:, :, slice]>mask_thresh & np.isfinite(m_img[:, :, slice])).reshape(IJ, 1)
  


        if slice == 0:
            temp1 = np.logical_not(nask[edges_start1])
            # print(temp1)
            # print(~ temp1)
            # print((temp1.reshape(-1, 1)).shape)
            # print(nask.shape)
            # print(mask[edges_start2]>mask_thresh & np.isfinite(mask[edges_start2]))
            # print((~nask[edges_start1]).shape)

            #print(((mask[edges_start2]>mask_thresh).reshape(-1, 1)).shape) #right shape for this
            # print(((np.isfinite(mask[edges_start2])).reshape(-1,1)).shape) #this is also right
            #print((((mask[edges_start2]>mask_thresh).reshape(-1, 1)) & ((np.isfinite(mask[edges_start2])).reshape(-1,1))).shape) #also good

            # print(((temp1.reshape(-1, 1)) & ((mask[edges_start2]>mask_thresh).reshape(-1, 1)) & ((np.isfinite(mask[edges_start2])).reshape(-1,1))).shape)
            surf = (temp1.reshape(-1, 1)) & ((mask[edges_start2]>mask_thresh).reshape(-1, 1)) & ((np.isfinite(mask[edges_start2])).reshape(-1,1))
            surf = np.array(np.where(surf))

            m0 = edges_start1[surf]
            m1 = edges_start2[surf]
        else:
            # print(slice, end = '')
           
            temp1 = np.logical_not(nask[edges_start1])
            # print((nask[edges_start1]).shape)
            # print(temp1.shape)
            surf = (temp1.reshape(-1, 1)) & ((mask[edges_start2]>mask_thresh).reshape(-1, 1)) & ((np.isfinite(mask[edges_start2])).reshape(-1,1))
            print(surf.shape)
            surf = np.array(np.where(surf))



            # SURF SHOULD BE 5x1 BUT IT IS 2x0!!!!!!!!!!

            # print(edges1.shape)
            # print(surf.shape)

            m0 = edges1[surf]

            m1 = edges2[surf]

        if surf.size != 0:
            dm = mask[m1] - mask[m0]
            mt = mask_thresh * np.ones(len(surf), 1)
            w = (mt -mask[m0])/(dm+(dm <=0)) * (dm>0)
            u0 = u[m0, :]
            u1 = u[m1, :]

            for i in range(n):
                v[m0, i] = (u0[:, i] * (1-w) + u1[:, i] * w)

            nask[m0] = 1

        if slice > 0:
            alteration2 = (2-flip)*IJ

            temp2 = ((dd_img[:, :, slice-1, :]).reshape(I,J,1,n))
            temp3 = ((v[np.arange(IJ) + alteration2, :]).reshape(I, J, 1, n))
            temp3 = temp2

            mm_img[:, :, slice -1] = (nask[np.arange(IJ) + alteration2]).reshape(I, J)

        if slice == numslices:
            alteration3 = (flip-1)*IJ
            temp4 = ((dd_img[:, :, slice, :]).reshape(I,J,1,n))
            temp5 = ((v[np.arange(IJ) + alteration3, :]).reshape(I, J, 1, n))
            temp4 = temp5

            mm_img[:, :, slice] = (nask[np.arange(IJ) + alteration3]).reshape(I, J)

print('Done\n')

# nib.save(dd, 'outputmesh.nii')
# nib.save(mm, 'ouputmask.nii')


mask_mesh('mask_coord.nii.gz', 'mask', 'mask.nii.gz')