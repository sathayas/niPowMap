from scipy.special import gammaln, gamma
import math
# 
# pm_Exp_ncX.py
# Non-central chi-square moments around 0.
#
# FORMAT:  mr = pm_Exp_NCX(df, delta, r)
# 
# PARAMETERS:
#    df:    degrees of freedom (a positive number)
#    delta: non-centrality parameter (a real number)
#    r:     the power (a positive number)
# Reference PowerMap/pm_Exp_ncX.m - https://sourceforge.net/projects/powermap/

def pm_Exp_ncX(df, delta, r):
    
    eps = 2.2204 * 10 ** -16
    tol = eps ** (7 / 8)

    j = 0
    isum = 0
    iisum = 1
    gamprod = 1
    if r < 0 and abs(r) % 1 < eps:
        bNegInt = 1
    else:
        bNegInt = 0

    while abs(iisum) > tol:
        gam = 1 / gamma(j + 1)
        d = math.exp(gammaln(r + j + (df / 2)) - gammaln(j + (df / 2))) * (delta / 2) ** j

        iisum = gam * d

        isum = isum + iisum

        j = j + 1

    mr = 2 ** r * math.exp(-delta / 2) * isum
    return mr