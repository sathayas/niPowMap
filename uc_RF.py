import math
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson
from scipy.special import gammaln, gamma
from scipy.stats import t as T
from scipy.stats import chi2
from scipy.stats import f
from scipy.stats import norm
import operator

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


def pm_ECdensity(STAT, t, df):
    EC = [0,0,0,0]

    if STAT == 'Z':

        a = 4 * math.log(2)
        b = math.exp(-t**2/2)

        EC[0] = 1 - norm.cdf(t)
        EC[1] = (a ** (1 / 2)) / (2 * math.pi) * b
        EC[2] = a/((2*math.pi)**(3/2))*b*t
        EC[3] = a**(3/2)/((2*math.pi)**2)*b*(t**2 - 1)

    elif STAT == 'T':
        v = df[1]
        a = 4 * math.log(2)
        b = math.exp(gammaln((v + 1) / 2) - gammaln(v / 2))
        c = np.power((1 + np.power(t, 2) / v), ((1 - v) / 2))

        EC[0] = 1 - T.cdf(t,v)
        EC[1] = (a ** (1 / 2)) / (2 * math.pi) * c
        EC[2] = np.multiply(a / (2 * math.pi) ** (3 / 2) * c, t / ((v / 2) ** (1 / 2)) * b)
        EC[3] = np.multiply(a ** (3 / 2) / ((2 * math.pi) ** 2) * c, ((v - 1) * (np.power(t, 2)) / v - 1))

    elif STAT == 'X':
        v = df[1]
        a = (4 * math.log(2)) / (2 * math.pi)
        b = np.multiply(np.power(t, 1 / 2 * (v - 1)), math.exp(-t / 2 - gammaln(v / 2)) / 2 ** ((v - 2) / 2))

        EC[0] = 1 - chi2.cdf(t, v)
        EC[1] = a ** (1 / 2) * b
        EC[2] = np.multiply(a * b, (t - (v - 1)))
        EC[3] = np.multiply(a ** (3 / 2) * b, (np.power(t, 2) - (2 * v - 1) * t + (v - 1) * (v - 2)))

    elif STAT == "F":
        k = df[0]
        v = df[1]
        a = (4 * math.log(2)) / (2 * math.pi)
        b = gammaln(v / 2) + gammaln(k / 2)

        EC[0] = 1 - f.cdf(t, df[0], df[1])
        EC[1] = a**(1/2)*math.exp(gammaln((v+k-1)/2)-b)*2**(1/2)*(k*t/v)**(1/2*(k-1))*(1+k*t/v)**(-1/2*(v+k-2))
        EC[2] = a*math.exp(gammaln((v+k-2)/2)-b)*(k*t/v)**(1/2*(k-2))*(1+k*t/v)**(-1/2*(v+k-2))*((v-1)*k*t/v-(k-1))
        EC[3] = a**(3/2)*math.exp(gammaln((v+k-3)/2)-b)*2**(-1/2)*(k*t/v)**(1/2*(k-3))*(1+k*t/v)**(-1/2*(v+k-2))*((v-1)*(v-2)*(k*t/v)**2-(2*v*k-v-k-1)*(k*t/v)+(k-1)*(k-2))

    return (np.asarray(EC).T)
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

u= uc_RF(0.05,[1,28], 'F', [47, -109.416922663253,	15495.5776301671, 288678.000000000],1)
print(u)