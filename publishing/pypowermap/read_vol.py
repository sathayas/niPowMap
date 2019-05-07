import nibabel as nib

def read_vol(fName):

	img = nib.load(fName)
	data = img.get_data()
	header = img.header

	return(header, data)