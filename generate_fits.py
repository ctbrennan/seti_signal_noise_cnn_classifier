from blimpy import Waterfall
import itertools
import numpy as np
from astropy.io import fits
import shutil
import os
"""
Had to change __setup_freqs in waterfall.py
"""

part_subdirectory = "./generated_fits/"

def pairwise(iterable):
	#Taken from https://docs.python.org/2/library/itertools.html
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def get_f_iter_bounds(header, frequency_dimension):
	n_channels_in_file  = header['nchans']
	if header['foff'] < 0 :
	    f_end  = header['fch1']
	    f_begin  = f_end + n_channels_in_file*header['foff']
	else:
	    f_begin  = header['fch1']
	    f_end  = f_begin + n_channels_in_file*header['foff']
	f_delta = frequency_dimension * header['foff']
	if f_delta <= 0:
		f_delta *= -1
	return f_begin, f_end, f_delta

def split_file(filename, frequency_dimension=512, count = None):
	wf = Waterfall(filename, load_data = False)
	f_begin, f_end, f_delta = get_f_iter_bounds(wf.header, frequency_dimension)
	
	parts = []
	for f_start, f_stop in pairwise(np.arange(f_begin, f_end, f_delta)):
		wf.read_data(f_start=f_start, f_stop = f_stop)
		parts.append(wf.data.squeeze()) #squeeze brings shape from (16, 1, 512) -> (16, 512)
		if count and len(parts) >= count:
			break
	return parts

def create_fresh_subdir(subdir):
	shutil.rmtree(subdir)
	os.makedirs(subdir)


def write_parts_to_files(filename, parts):
	prefix = part_subdirectory + filename[2:-4] + "-" #currently just removing ./ and file extension fil
	for i, ndarray in enumerate(parts):
		part_filename = prefix + str(i) + ".fits"
		make_fits_from_ndarray(ndarray, part_filename)


def make_fits_from_ndarray(arr, filename):
	hdu = fits.PrimaryHDU(arr)
	hdu.writeto(filename)

def main():
	filename = "./spliced_blc0001020304050607_guppi_57550_40640_GJ1002_0003.gpuspec.0000.fil"
	create_fresh_subdir(part_subdirectory)
	parts = split_file(filename, count = 50)
	write_parts_to_files(filename, parts)


if __name__ == "__main__":
	main()