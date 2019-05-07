import nibabel as nib

# Purpose   Wrapper function to read a 3D or 4D analyze or nifti image
#           using the nibabel
#
# Inputs    fName   - File name of analyze or nifi image
#
# Outputs   header       - Image header
#           data      - Image data

def read_vol(fName):

	img = nib.load(fName)
	data = img.get_data()
	header = img.header

	return(header, data)