import numpy as np
import nibabel as nib
import sys
import os


def est_fwhm(x, df, stat):
    '''
    Function est_fwhm.py
    Calculates smoothness of a statistic image (T- or F-statistic) directly 
    from a statistic image rather than from residuals.

    input parameters:
          x:        A 3D matrix of statistic image data.
          df:       A 1x2 vector of degrees of freedom, with [df1 df2]. For an
                    F-image, df1 and df2 correspond to the numerator and demoninator
                    degrees of freedom, respectively. For a T-image, df1=1 and
                    df2 is the error df.
          stat:     Statistical image type
                    'T' - T-image
                    'F' - F-image

    returns:
          fwhm:     A 1x3 vector of FWHM in terms of voxels, in x, y, and z
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

    # First, checking the input
    if stat=='T' and np.isscalar(df):
        df1 = 1
        df2 = df
    else:
        df1, df2 = df

    dfsum = df1 + df2

    if (df2<=4) or ((stat=='F') and (df2<=6)):
        print('Degrees of freedom is too small!')
        return

    
    # defining necessary parameters for calculation
    #------------------------------------------------------------------------------------

    tol = 0.000000000001; # a tiny tiny value

    xdim   = size(x,1);
    ydim   = size(x,2);
    zdim   = size(x,3);

    



