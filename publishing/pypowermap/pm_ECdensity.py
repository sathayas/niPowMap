import math
import numpy as np
from scipy.special import gammaln
from scipy.stats import t as T
from scipy.stats import chi2
from scipy.stats import f
from scipy.stats import norm

def pm_ECdensity(STAT, t, df):
    # returns ECdensities
    # Reference: PowerMap/pm_P_RF.m - https://sourceforge.net/projects/powermap/
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