from publishing.pypowermap.P_RF import P_RF
from publishing.pypowermap.pm_u import pm_u

# corrected critical height threshold at a specified significance level
# FORMAT u = uc_RF(a,df,STAT,R,n)
# a     - critical probability - {alpha}
# df    - [df{interest} df{residuals}]
# STAT  - Statistical field
#		'Z' - Gaussian field
#		'T' - T - field
#		'X' - Chi squared field
#		'F' - F - field
# R     - RESEL Count {defining search volume}
# n     - number of conjoint SPMs
#
# u     - critical height {corrected}
# Reference PowerMap/pm_uc_RF.m - https://sourceforge.net/projects/powermap/

def uc_RF(a, df, STAT, R, n):

    u = pm_u((a/sum(R))**(1/n), df,STAT)
    du = 1e-6

    d = 1

    while abs(d)>1e-6:

        [P,P,p,En,EN] = P_RF(1, 0, u, df, STAT, R, n)
        [P,P,q, En, EN] = P_RF(1, 0, u+du, df, STAT, R, n)
        d = (a-p)/((q-p)/du)
        u = u+d

    d = 1

    while abs(d) > 1e-6:
        [P, P, p, En, EN] = P_RF(1, 0, u, df, STAT, R, n)
        [P, P, q, En, EN] = P_RF(1, 0, u + du, df, STAT, R, n)
        d = (a-p)/((q-p)/du)
        u = u+d


    return (u)