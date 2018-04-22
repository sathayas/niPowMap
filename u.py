from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import f

def u (a, df, STAT):

	if STAT == 'Z':
		u = norm.ppf(0.95)

	elif STAT == 'T':
		u = t.ppf(1-a, df(2))

	elif STAT == 'X':
		u = chi2.ppf(1-a, df(2))

	elif STAT == 'F':
		u = f.ppf(1-a, df(1), df(2))

	elif STAT == "P":
		u = a
	return (u)




