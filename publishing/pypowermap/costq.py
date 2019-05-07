from scipy.stats import t
from scipy.stats import nct
from scipy.stats import f

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