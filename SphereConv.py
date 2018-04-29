def SphereConv(P, Q, r):
	tol = np.finfo(float).eps
	tol = tol**(7/8)

	if (isinstance(P, str)):
		hdrP, P = read_vol(P)


	#isstruct not exist in python
	if (hdrP):
		VOX = hdrP['pixdim'][1:4]

		if VOX[0] < 1:
			flipdimFlag = 2
		elif VOX[1] <1:
			flipdimFlag = 1
		elif VOX[2] < 1:
			flipdimFlag = 3
		else:
			flipdimFlag = 0

		VOX = abs(VOX)

	else:
		VOX = [1, 1, 1]

	hdrXin = hdrP
	Xin = P
	oringinalshapetuple= Xin.shape

	if flipdimFlag > 0:
		Xin = np.flip(Xin, flipdimFlag-1)

	bXin = np.ones(Xin.shape)
	bXin = bXin.reshape(1,bXin.size)
	Xin = Xin.reshape(1,bXin.size)
	tmp = np.argwhere(np.isnan(Xin))
	tmp = tmp[:, 1]
	tmp2 = np.argwhere(np.isfinite(Xin))
	tmp2 = tmp2[:, 1]


	for x in np.array(tmp).flat:
		Xin[0,x] = 0
		bXin[0, x] = 0

	r2 = np.divide(r, VOX*2)
	r2ones = np.ones(r2.shape)
	r2 = np.maximum(r2, r2ones)

	rdim = 2*r2+1
	roff = np.multiply(r2+1, VOX*2)

	x = np.arange(1,rdim[0])
	y = np.arange(1,rdim[0])
	z = np.arange(1,rdim[0])

	x, y, z = np.meshgrid(x, y, z)
	
	x = x*(VOX[0]*2)
	x = np.transpose(x, (0, 2, 1))
	y = y*(VOX[1]*2)
	y = np.transpose(y, (1, 0, 2))
	z = z*(VOX[2]*2)
	z = np.transpose(z, (2, 1, 0))

	xrSphtemp = np.power(x-roff[0], 2)
	yrSphtemp = np.power(y-roff[1], 2)
	zrSphtemp = np.power(z-roff[2], 2)
	Sphtemp = xrSphtemp+yrSphtemp+zrSphtemp

	rSph = np.power(Sphtemp, 1/2)
	brSph = rSph<(r+tol)

	sumSph = sum(brSph[:])
	sumSph = (sum(sum(sumSph)))

	Xout = np.convolve(np.ravel(Xin), np.ravel(brSph), 'same')
	Xout = np.reshape(Xout, oringinalshapetuple)

	sumXout = np.convolve(np.ravel(bXin), np.ravel(brSph), 'same')
	sumXout = np.reshape(sumXout, oringinalshapetuple)
	Xout = np.ravel(Xout)

	x1 = np.divide(Xout[tmp2], sumXout[tmp2])
	Xout[tmp2] = np.divide(Xout[tmp2], sumXout[tmp2]) 
	XOut[tmp] = np.nan

	pm_write_vol(hdrXin, Xout, Q)





