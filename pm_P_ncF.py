import math
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson, ncf
from scipy.special import gammaln, gamma
import operator

def P_ncF(s, df1, df2, delta, R):
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


def pm_Exp_ncX(df, delta, r):
    eps = 2.2204*10**-16
    tol = eps ** (7 / 8)
    
    j = 0
    isum = 0
    iisum = 1
    gamprod = 1
    if r < 0 and abs(r)%1 < eps:
        bNegInt = 1
    else:
        bNegInt = 0
    
    while abs(iisum) > tol:
    
        gam = 1 / gamma(j + 1)
        d = math.exp(gammaln(r + j + (df / 2)) - gammaln(j + (df / 2))) * (delta / 2) ** j
    
        iisum = gam *d
    
        isum = isum + iisum
    
        j = j + 1
    
    mr = 2 ** r * math.exp(-delta / 2) * isum
    return mr

a = P_ncF(40,2,20,2.5,[1, 4, 2*math.pi, (4/3)*math.pi])
print(a)