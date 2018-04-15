import math 
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import poisson 

def P_ncF(s, df1, df2, delta, R):
	k = 0
	n = 1
	c = 1

	D = [index for index, item in enumerate(R) if item != 0][-1]
	R = R[:D+1]

	arange = np.arange(D)
	temp = np.random.gamma(arange/2)
	G =np.divide((math.sqrt(math.pi)), temp)
	# external function
	EC = ECncF(s, df1, df2, delta)
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
	beta = (np.random.gamma(D/2 +1)/En)**(2/D)
	p = exp(-beta*(k**(2/D)))
	P = 1- poisson.cdf(c-1, (Em + eps)*p)

	return (P, Emm, En, EN)