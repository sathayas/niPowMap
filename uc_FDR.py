import numpy as np
from scipy.stats import t
from scipy.stats import nct
from scipy.stats import f
from scipy.stats import ncf

def uc_FDR(q,df, STAT, nc, Vall, Vsig):
	tol = 0.000000001
	psig = Vsig/Vall

	k1 = 0.0001
	if STAT == 'T':
		k2 = 50.1

	elif STAT == 'F':
		k2 = 500

	dq = k2-k1

	while (dq >tol):

		km = (k1+k2)/2
		f1 = costq(k1, q, df, nc, psig, STAT)
		fm  = costq(km,q,df,nc,psig,STAT)
		f2  = costq(k2,q,df,nc,psig,STAT)

		if f1*fm <0:
			k2 = km
		elif f2*fm < 0:
			k1 = km
		else:
			u = np.NaN
			return 

		dq = abs(k2 -k1)

	if STAT == "T":
		F0 = t.cdf(km, df(2))
		F1 = nct.cdf(km, df(2), nc)

	elif STAT == "F":
		F0 = f.cdf(km, df(1), df(2))
		F1 = ncf.cdf(km, df(1), df(2), nc)

	obsq = (1-psig) * (1-F0)/((1-psig)*(1-F0) + psig*(1-F1))

	if abs(obsq -q) >0.0001:
		u = np.NaN
	else:
		u = km

	return (u)

def costq(u, q, df, nc, psig, STAT):

	tol     = 0.000000001
	if STAT == 'T':
		F0 = t.cdf(u, df(2))
		F1 = nct.cdf(u, df(2), nc)

	elif STAT == 'F':
		F0 = f.cdf(u, df(1), df(2))
		F1 = nct.cdf(u, df(1), df(2), nc)

	if (F0>1-tol) & (1>1-tol) & (abs(F0-F1)<tol):
		F = -q*tol
	else:
		a1 = (1-psig)*(1-F0)
		a2 = (1-psig) * (1-F0)+psig*(1-F1)
		F = a1 - q*a2

	return (F)