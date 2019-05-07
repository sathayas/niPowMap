import nibabel as nib
import numpy as np
def mesh_tet(input_file, output_file_base):

    #MESH_TET finds lengths of edges of a tetrahedral mesh
    #
    # Originally written for fmristat package by Keith Worsley.
    #
    # This is a modified version using read/write functions from nifti toolbox.
    # There are other modifications to make the program shorter and run faster.
    # Reference PowerMap/mesh_tet.m - https://sourceforge.net/projects/powermap/
    #-----------------------------------------------------------------------------------
    #
    
    
    # Initialization
    #-----------------------------------------------------------------------------------
    mask_thresh   = 0


    #-loading in the mesh file
    #-----------------------------------------------------------------------------------
    d          = nib.load(input_file)
    dime       = d.header['dim']
    n          = dime[4]
    numslices  = dime[3]
    J          = dime[2]
    I          = dime[1]
    
    
    #-Organizing output file
    #-----------------------------------------------------------------------------------
    #-organizing the output (tetrahedron)
    lam        = d  #-recycling the header info from the mesh file
    #lam.fileprefix        = [deblank(output_file_base) '_tet']
    lam.header['data_type']      = 16
    lam.header['bitpix']       = 32
    lam.header['dim'][1:5] = [I, J, numslices, 6]
    lam_img    = np.zeros(lam.header['dim'][1:5])
    
    
    
    # Set up:
    #-----------------------------------------------------------------------------------
    
    i     = np.kron(np.ones((1,J)),np.arange(1,I+1))
    j     = np.kron(np.arange(1,J+1),np.ones((1,I)))
    
    IJ    = I*J
    ex    = np.where(i<I)[1]
    ex1   = np.array([ex+1, ex+IJ+1]).transpose()
    ex2i = np.where(i>1)[1]+1
    ex2j = np.where(i>1)[1]+IJ+1
    ex2   = np.array([ex2i, ex2j]).transpose()
    
    ey    = np.where(j<J)[1]
    ey1   = np.array([ey+1, ey+IJ+1]).transpose()
    ey2   = np.array([np.where(j>1)[1]+1, np.where(j>1)[1]+IJ+1]).transpose()
    
    ez    = np.arange(1,IJ+1)
    ez1   = ez
    ez2   = ez+IJ
    
    exye  = np.where(((i+j)%2 ==0) & (i<I) & (j<J))[1]+1
    exyo  = np.where(((i+j)%2 ==1) & (i<I) & (j<J))[1]+1
    exy   = np.concatenate([exye, exyo])
    exy1  = np.array([np.concatenate([exye,exyo+1]), np.concatenate([exye+1+IJ,exyo+IJ])]).transpose()
    exy2 = np.array([np.concatenate([exye+1+I, exyo+I]), np.concatenate([exye+I+IJ, exyo+1+I+IJ])]).transpose()
    
    exze  = np.where(((i+j)%2 ==0) & (i<I))[1] +1
    exzo  = np.where(((i+j)%2 ==1) & (i<I))[1] +1
    exz   = np.concatenate([exze, exzo])
    exz1  = np.concatenate([exze, exzo+1])
    exz2  = np.concatenate([exze+1+IJ, exzo+IJ])
    
    eyze  = np.where(((i+j)%2 ==0) & (j<J))[1] +1
    eyzo  = np.where(((i+j)%2 ==1) & (j<J))[1] +1
    eyz   = np.concatenate([eyze, eyzo])
    eyz1  = np.concatenate([eyze, eyzo+I])
    eyz2  = np.concatenate([eyze+I+IJ, eyzo+IJ])
    
    
    
    # START:
    #-----------------------------------------------------------------------------------
    
    lams  = np.zeros((1,IJ*9))
    u     = np.zeros((2*IJ,n))
    flip  = 1
    print('Calculating tetrahedral mesh')
    for slice in range(1,numslices+1):
        print('.')
        flip   = 3-flip
        tmpimg = d.get_data()[:,:,slice-1,0:n]
        u[np.arange(0,IJ)+(flip-1)*IJ,:] = np.reshape(tmpimg,(IJ,n), order= 'F')
    
        start  = ((flip-1)*6*IJ)
        lams[0,start + ex]    = np.sum((u[ex1[:,flip-1]-1,:]-u[ex2[:, flip-1]-1,:])**2, 1)
        lams[0,start+  IJ+ ey]    = np.sum((u[ ey1[:,flip-1]-1,:]-u[ ey2[:,flip-1]-1,:])**2,1)
        lams[0,start+2*IJ+exy]   = np.sum((u[ exy1[:,flip-1]-1,:]-u[ exy2[:,flip-1]-1,:])**2,1)
        lam_img[:,:,slice-1,0:3]  = np.reshape(lams[0,(flip-1)*6*IJ + np.arange(0,(3*IJ))],(I,J,3))
        if slice>1:
            lams[0,3*IJ+ ez]      = np.sum((u[ez1-1 ,:]-u[ez2-1 ,:])**2,1)
            lams[0,4*IJ+exz]      = np.sum((u[exz1-1,:]-u[exz2-1,:])**2,1)
            lams[0,5*IJ+eyz]      = np.sum((u[eyz1-1,:]-u[eyz2-1,:])**2,1)
            lam_img[:,:,slice-1,3:6] = np.reshape(lams[0,3*IJ+np.arange(0,(3*IJ))],(I,J,3))

    print('Done!\n')
    
    print(lam_img)
    #-writing out the tetrahedral mesh file
    #-----------------------------------------------------------------------------------
    output = nib.Nifti1Image(lam_img, d.affine)
    nib.save(output, output_file_base)

    
    return

mesh_tet('/testdata/mask_coord.nii.gz', '/testdata/tet.nii.gz')
