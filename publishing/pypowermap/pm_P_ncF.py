import math
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson
from scipy.special import gammaln, gamma
import operator
from publishing.pypowermap.pm_ECncF import pm_ECncF

#
# Calculates the corrected p-value for a non-central F-image
#
# Usage: [P,Em,En,EN] = pm_P_ncF(s,df1,df2,delta,R)
# Parameters:
#       s:      The value of a non-central F random variable at which EC 
#               is calculated.
#       df1:    Numerator degrees of freedom
#       df2:    Denominator degrees of freedom
#       delta:  Non-centrality parameter
#       R:      Resel counts for the search volume
#       P:      Corrected p-value (FWE-corrected)
#       Em:     Expected number of clusters
#       En:     Expected number of voxels per cluster
#       EN:     Expected number of suprathreshold voxels
#__________________________________________________________________________
# Reference PowerMap/pm_P_ncF.m - https://sourceforge.net/projects/powermap/

def pm_P_ncF(s, df1, df2, delta, R):
    eps = 2.2204 * 10 ** -16
    k = 0
    n = 1
    c = 1

    D, value = max(enumerate(R), key=operator.itemgetter(1))
    R = R[:D+1]

    arange = np.arange(D+1)
    temp = gamma((arange+1) / 2)
    G = np.divide((math.sqrt(math.pi)), temp)
    EC = pm_ECncF(s, df1, df2, delta)
    EC = EC[:D+1]+ eps
    #EC = np.ndarray(EC)

    temp2 = np.multiply(EC, G)
    temp3 = toeplitz(temp2)

    P = (np.triu(temp3)) ** n
    P = P[0, :]
    EM = (R/G)*P
    Em = sum(EM)
    EN = P[0] * R[D]
    En = EN/EM[D]

    D = D-1
    beta = (gamma(D/2 +1)/En)**(2/D)
    p = math.exp(-beta*(k**(2/D)))
    P = 1- poisson.cdf(c-1, (Em + eps)*p)

    return [P, Em, En, EN]