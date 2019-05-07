from scipy.stats import t as T
from scipy.stats import chi2
from scipy.stats import f
from scipy.stats import norm

# uncorrected critical height threshold at a specified significance level
# FORMAT [u] = spm_u(a,df,STAT)
# a     - critical probability - {alpha}
# df    - [df{interest} df{error}]
# STAT  - Statistical field
#               'Z' - Gaussian field
#               'T' - T - field
#               'X' - Chi squared field
#               'F' - F - field
#
# u     - critical height {uncorrected}
# Reference PowerMap/pm_u.m - https://sourceforge.net/projects/powermap/

def pm_u(a,df,STAT):
    if STAT == 'Z':
        return norm.ppf(1-a)

    elif STAT == 'T':
        return T.ppf(1-a, df[1])

    elif STAT == 'X':
        return chi2.ppf(1-a,df[1])

    elif STAT == "F":
        return f.ppf(1-a,df[0],df[1])

    elif STAT == "P":
        return a