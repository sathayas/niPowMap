def uc_RF(a, df, STAT, R, n):

	u = pm_u((a/sum(R))**(1/n), df, STAT)
	du = 1e-6

	d = 1

	while abs(d)>1e-6:

		# P_RF(1, 0, u, df, STAT, R, n)
		#P_RF(1, 0, u+df, df, STAT, R, n)
		d = (a-p)/((q-p)/du)
		u = u+d

	d = 1

	while abs(d) > 1e-6:
		p = P_RF(1, 0, u, df, STAT, R, n)
		q = P_RF(1, 0, u + df, df, STAT, R, n)
		d = (a-p)/((q-p)/du)
		u = u+d


	return (u)
