import numpy as np 
def SphereConv(P, Q, r):
	#in the actual matlab its eps(7/8) but I can't find comparable func in python
	tol = np.finfo.eps

	if (isinstance(P, str)):
		hdr, P = pm_read_vol(P)

	#isstruct not exist in python
	if (true):
		VOX = hdrP.dime.pixdim[1:4]

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

	if flipdimFlag > 0:
		Xin = np.flip(Xin, flipdimFlag-1)

	bXin = np.ones(Xin.shape)
	tmp = np.argwhere(np.isnan(Xin))
	tmp2 = np.argwhere(np.isfinite(Xin))

	Xin[tmp] = 0
	bXin[tmp] = 0

	r2 = np.divide(r, VOX)
	r2ones = ones(r2.shape)
	r2 = np.maximum(r2, r2ones)

	rdim = 2*r2+1
	roff = np.multiply(r2+1, VOX)

	x = meshgrid[0:rdim[0]]
	y = meshgrid[0:rdim[1]]
	z = meshgrid[0:rdim[2]]

	x = x*VOX[0]
	y = y*VOX[1]
	z = z*VOX[2]

	xrSphtemp = np.power(x-roff[0], 2)
	yrSphtemp = np.power(y-roff[1], 2)
	zrSphtemp = np.power(z-roff[2], 2)
	Sphtemp = xrSphtemp+yrSphtemp+zrSphtemp

	rSph = np.power(Sphtemp, 1/2)

	brSph = rSph<(r+tol)
	sumSph = sum(brSph)

	Xout = convn(Xin, brSph, 'same')
	sumXout = convn(bXin, brSph, 'same')

	XOut[tmp2] = np.divide(XOut[tmp2], sumXout[tmp2])
	XOut[tmp] = np.nan

	pm_write_vol(hdrXin, Xout, Q)











