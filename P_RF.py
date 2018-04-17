import math 
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson 
from scipy.special import gammaln
from scipy.stats import t
from scipy.stats import chi2

def P_RF(c, k, z, df, STAT, R, n):

	D = [index for index, item in enumerate(R) if item != 0][-1]
	R = R[:D+1]

	arange = np.arange(D)
	temp = np.random.gamma(arange/2)
	G =np.divide((math.sqrt(math.pi)), temp)
	EC = pm_ECdensity(STAT, z, df)
	EC = EC[arange] + eps
	EC = numpy.ndarray(EC)

	temp2 = np.multiply(EC.T, G)
	temp3 = toeplitz(temp2)

	P = (np.triu(temp3))**n
	p = P[0, :]
	EM = np.multiply((np.divide(R, G)), P)
	Em = sum(EM)
	EN = P[0] * R[D]
	En = EN/EM[D]

	D = D-1

	if (not k or not D):
		p = 1

	elif STAT == 'Z':
		beta = (np.random.gamma(D/2 +1)/En)**(2/D)
		p = exp(-beta*(k**(2/D)))

	elif STAT == 'T':
		beta = (np.random.gamma(D/2 +1)/En)**(2/D)
		p = exp(-beta*(k**(2/D)))

	elif STAT == 'X':
		beta = (np.random.gamma(D/2 +1)/En)**(2/D)
		p = exp(-beta*(k**(2/D)))

	elif STAT == 'F':
		beta = (np.random.gamma(D/2 +1)/En)**(2/D)
		p = exp(-beta*(k**(2/D)))

	P = 1- poisson.cdf(c-1, (Em + eps)*p)

	if (k>0 and n>1):
		P = np.array()
		p = np.array()
	if (k>0 and (STAT == 'X' or STAT == 'F')):
		P = np.array()
		p = np.array()

	return (P, p, Em, En, EN)

def pm_ECdensity(STAT, t1, df):
	t1 = t1.transpose()
	if STAT == 'Z':
		
		a = 4*log(2)
		b = exp((numpy.power(-t, 2))/2)

		EC[0, :] = 1- norm.cdf(t)
		EC[1, :] = (a**(1/2))/(2*math.pi)*b 
		EC[2, :] = numpy.multiply(a/((2*math.pi)**(3/2)*b), t1)
		EC[3, :] = numpy.multiply(a**(3/2)/ ((2*math.pi)**2)*b, (numpy.power(t1, 2)-1))

	elif STAT == 'T':
		v =  df(2)
		a = 4*log(2)
		b = exp(gammaln((v+1)/2) - gammaln(v/2))
		c = numpy.power((1+numpy.power(t,2)/v), ((1-v)/2))

		EC[0, :] = 1 - t.cdf(t1,v)
		EC[1, :] = (a**(1/2))/(2*math.pi)*c
		EC[2, :] = numpy.multiply(a/(2*math.pi)**(3/2)*c, t1/((v/2)**(1/2))*b)
		EC[3, :] = numpy.multiply(a**(3/2)/((2*math.pi)**2)*c, ((v-1)*(numpy.power(t, 2))/v -1))

	elif STAT == 'X':
		v = df(2)
		a = (4*log(2))/(2*math.pi)
		b = numpy.multiply(numpy.power(t, 1/2*(v-1)), exp(-t1/2 - gammaln(v/2))/2**((v-2)/2))

		EC[0, :] = 1- chi2.cdf(t,v)
		EC[1, :] = a*(1/2)*b
		EC[2, :] = numpy.multiply(a*b, (t-(v-1)))
		EC[3, :] = numpy.multiply(a**(3/2)*b, (numpy.power(t, 2)-(2*v-1)*t+(v-1)*(v-2)))
		


	return (EC)