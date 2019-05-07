import math
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson
from scipy.special import gammaln, gamma
import operator
from publishing.pypowermap.pm_ECdensity import pm_ECdensity

# Returns the [un]corrected P value using unifed EC theory
# FORMAT [P p Em En EN] = P_RF(c,k,Z,df,STAT,R,n)
#
# c     - cluster number 
# k     - extent {RESELS}
# Z     - height {minimum over n values}
# df    - [df{interest} df{error}]
# STAT  - Statistical field
#		'Z' - Gaussian field
#		'T' - T - field
#		'X' - Chi squared field
#		'F' - F - field
# R     - RESEL Count {defining search volume}
# n     - number of component SPMs in conjunction
#
# P     - corrected   P value  - P(n > kmax}
# p     - uncorrected P value  - P(n > k}
# Em    - expected total number of maxima {m}
# En    - expected total number of resels per cluster {n}
# EN    - expected total number of voxels {N}
#
#___________________________________________________________________________
# Reference PowerMap/pm_P_RF.m - https://sourceforge.net/projects/powermap/
# Ref: Hasofer AM (1978) Upcrossings of random fields
# Suppl Adv Appl Prob 10:14-21
# Ref: Friston et al (1993) Comparing functional images: Assessing
# the spatial extent of activation foci
# Ref: Worsley KJ et al 1996, Hum Brain Mapp. 4:58-73

def P_RF(c, k, z, df, STAT, R, n):
    eps = 2.2204*10**-16
    D, value = max(enumerate(R), key=operator.itemgetter(1))
    R = R[:D+1]

    arange = np.arange(D+1)
    temp = gamma((arange+1) / 2)
    G = np.divide((math.sqrt(math.pi)), temp)
    EC = pm_ECdensity(STAT, z, df)
    EC = EC[:D+1] + eps

    temp2 = np.multiply(EC.T, G)
    temp3 = toeplitz(temp2)

    P = (np.triu(temp3)) ** n
    P = P[0, :]
    EM = np.multiply((np.divide(R, G)), P)
    Em = sum(EM)
    EN = P[0] * R[D]
    En = EN / EM[D]

    D = D - 1

    if (not k or not D):
        p = 1

    elif STAT == 'Z':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    elif STAT == 'T':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    elif STAT == 'X':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    elif STAT == 'F':
        beta = (gamma(D / 2 + 1) / En) ** (2 / D)
        p = math.exp(-beta * (k ** (2 / D)))

    P = 1 - poisson.cdf(c - 1, (Em + eps) * p)

    if (k > 0 and n > 1):
        P = np.array()
        p = np.array()
    if (k > 0 and (STAT == 'X' or STAT == 'F')):
        P = np.array()
        p = np.array()

    return (P, p, Em, En, EN)
