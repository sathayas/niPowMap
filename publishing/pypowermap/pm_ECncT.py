import math
import numpy as np
from scipy.stats import nct
from publishing.pypowermap.pm_Exp_ncX import pm_Exp_ncX

def pm_ECncT(s,df,delta):
    a = 4 * math.log(2)
    b = 2 * math.pi
    c = s **2 / df
    d = 1 + c

    tmpEC = np.zeros(4)
    tmpEC[0] = 1 - nct.cdf(s, df, delta)

    tmpEC[1] = a ** (1 / 2) * b ** (-1 / 2) * df ** (1 / 2) * d * pm_Exp_ncX(df + 1, delta ** 2, -1 / 2) * nct.pdf(s, df, delta)

    Q1 = a * b ** (-1) * df ** (1 / 2) * d
    Q2 = d ** (-1 / 2) * pm_Exp_ncX(df + 1, delta ** 2, -1 / 2) * delta
    Q3 = (df - 1) * c ** (1 / 2) * pm_Exp_ncX(df + 1, delta ** 2, -1)
    Q4 = nct.pdf(s, df, delta)
    tmpEC[2] = (-1) * Q1 * (Q2 - Q3) * Q4


    P1 = a ** (3 / 2) * b ** (-3 / 2) * df ** (1 / 2) * d
    P2 = (df - 1) * (df - 2) * c * pm_Exp_ncX(df + 1, delta ** 2, -3 / 2)
    P3 = 2 * (df - 1) * c ** (1 / 2) * d ** (-1 / 2) * pm_Exp_ncX(df + 1, delta ** 2, -1) * delta
    P4 = d ** (-1) * pm_Exp_ncX(df + 1, delta ** 2, -1 / 2) * delta ** 2
    P5 = pm_Exp_ncX(df + 1, delta ** 2, -1 / 2)
    P6 = nct.pdf(s, df, delta)
    tmpEC[3] = P1 * (P2 - P3 + P4 - P5) * P6

    return tmpEC