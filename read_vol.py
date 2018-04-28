def read_vol(fName):
	# example_filename = os.path.join(data_path, fName)
	directory, file = os.path.split(fName)
	filename_w_ext = os.path.basename(fName)
	file, ext = os.path.splitext(filename_w_ext)
	example_filename = os.path.join(directory, filename_w_ext)

	img = nib.load(example_filename)
	data = img.get_data()
	header = img.header

	return(header, data)
