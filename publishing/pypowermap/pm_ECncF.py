import math
import numpy as np
from scipy.stats import ncf
from publishing.pypowermap.pm_Exp_ncX import pm_Exp_ncX

#
# Euler characteristic for non-central F distribution.
#
# Usage: [EC] = pm_ECncF(s,df1,df2,delta)
# Parameters:
#       s:      The value of a non-central F random variable at which EC 
#               is calculated.
#       df1:    Numerator degrees of freedom
#       df2:    Denominator degrees of freedom
#       delta:  Non-centrality parameter (for the numerator)
#__________________________________________________________________________
# Reference: PowerMap/pm_ECncF.m - https://sourceforge.net/projects/powermap/

def pm_ECncF(s,n1,n2,delta):
    

    a        = 4*math.log(2)
    b        = 2*math.pi
    c        = n1*s/n2
    d        = 1+c

    tmpEC    = np.zeros(4)
    tmpEC[0] = 1-ncf.cdf(s, n1, n2, delta)

    tmpEC[1] = a**(1/2) * b**(-1/2) * 2 * d * c**(1/2) * pm_Exp_ncX(n1+n2, delta, -1/2) * (n2/n1) * ncf.pdf(s,n1,n2,delta)

    Q1       = a * b**(-1) * 2 * d
    Q2       = pm_Exp_ncX(n1+n2, delta, -1) * ((n1-1) - (n2-1)*c + delta)
    Q3       = (n2/n1) * ncf.pdf(s,n1,n2,delta)
    tmpEC[2] = (-1) * Q1 * Q2 * Q3

    P1       = a**(3/2) * b**(-3/2) * 2 * d * c**(-1/2)
    P2       = pm_Exp_ncX(n1+n2, delta, -3/2)
    P3       = (n1-1)*(n1-2) - 2*(n2-1)*(n1-1+delta)*c + (n2-1)*(n2-2)*c**2 + delta*(2*n1-1+delta)
    P4       = c*pm_Exp_ncX(n1+n2, delta, -1/2)
    P5       = (n2/n1) * ncf.pdf(s,n1,n2,delta)
    tmpEC[3] = P1 * (P2*P3 - P4) * P5

    return tmpEC