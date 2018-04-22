import math 
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson 
from scipy.special import gammaln
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import f

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
		EC[1, :] = a**(1/2)*b
		EC[2, :] = numpy.multiply(a*b, (t-(v-1)))
		EC[3, :] = numpy.multiply(a**(3/2)*b, (numpy.power(t, 2)-(2*v-1)*t+(v-1)*(v-2)))

	elif STAT == "F":
		k = df(1)
		v = df(2)
		a = (4*log(2))/(2*math.pi)
		b = gammaln(v/2) + gammaln(k/2)

		EC[0, :] = 1 - f.cdf(t, df(1), df(2))
		temp1 = (a**(1/2)*exp((gammaln(v+k-1)/2)-b)*2^(1/2)*(k*t/v))
		temp2 = numpy.power(temp1, (1/2*(k-1)))
		temp3 = numpy.multiple(temp2, (1+k*t/v))
		temp4 = numpy.power(temp3, (-1/2*(v+k-2)))
		EC[1, :] = temp4

		temp5 = a*exp(gammaln(v+k-2)/2)-b*(k*t/v)
		temp6 = numpy.power(temp5, (1/2*(k-2)))
		temp7 = numpy.multiply(temp6, (1+k*t/v))
		temp8 = numpy.power(temp7, -1/2*(v+k-2))
		temp9 = numpy.multiply(temp8, ((v-1)*k*t/v-(k-1)))
		EC[2, :] = temp9

		temp10 = a**(3/2)*exp(gammaln((v+k-3)/2)-b)*2**(-1/2)*(k*t/v)
		temp11= numpy.power(temp10, (1/2*(k-3)))
		temp12 = numpy.multiply(temp11, 1+k*t/v)
		temp13 = numpy.power(temp12, (-1/2*(v+k-2)))
		temp14 = numpy.multiply(temp13, ((v-1)*(v-2)*(k*t/v)))
		temp15 = numpy.power(temp14, 2-(2*v*k-v-k-1)*(k*t/v)+(k-1)*(k-2))

		EC[3, :] = temp15



	return (EC)