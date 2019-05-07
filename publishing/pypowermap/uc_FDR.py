import numpy as np
from scipy.stats import t
from scipy.stats import nct
from scipy.stats import f
from scipy.stats import ncf
from publishing.pypowermap.costq import costq

# False Discovery critical height threshold
# Calculated based on the assumption that a sphere-shaped signal is located
# in the middle of null voxels.
#
# FORMAT u = uc_FDR(q,df,effsz,STAT,Vtotal,Vsig)
#
# q     - critical expected False Discovery Rate
# df    - [df{interest} df{residuals}]
# STAT  - Statistical field
#		'T' - T - field for the null voxels and non-central T-field for the
#             signal voxels
#		'F' - F - field for the null voxels and non-central F-field for the
#             signal voxels
# nc    - The non-centrality parameter summarizing the strength of the
#         signal. In case of 'F', 
# Vall  - The total brain volume (either resels or voxels)
# Vsig  - The signal volume (it has to be in the same unit as Vtotal)
#
# u     - critical height
#
#___________________________________________________________________________
#
#
# This function calculates the FDR-corrected threshold theoretically. This
# is done by assuming the organization of the statistic image. In
# particular, the signal is confined to a subset of the brain volume,
# having volume Vsig, whereas the rest of the brain there is no signal. The
# signal is considered to follow a non-central distribution with
# non-centrality nc, whereas the remaining non-signal areas are considered
# to follow the central distribution.
#
# Reference: Reference PowerMap/pm_uc_FDR.m - https://sourceforge.net/projects/powermap/
#--------------------------------------------------------------------------
#

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
