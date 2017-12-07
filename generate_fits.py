#!/usr/bin/env python
from blimpy import Waterfall
import itertools
import numpy as np
from astropy.io import fits
import shutil
import os
"""
Had to change __setup_freqs in waterfall.py

read 1 GB at a time from waterfall file, then split that into however many .fits of dim 512, feed that data in one (probably one) or two parts into classifier
only generate file if classified as signal, pool along frequency dimension, make png

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
	return [part for part in split_file_generator(filename, frequency_dimension, count)]
	# wf = Waterfall(filename, load_data = False)
	# f_begin, f_end, f_delta = get_f_iter_bounds(wf.header, frequency_dimension)
	
	# parts = []
	# for f_start, f_stop in pairwise(np.arange(f_begin, f_end, f_delta)):
	# 	wf.read_data(f_start=f_start, f_stop = f_stop) # want to call read data fewer times, cut data into smaller pieces after
	# 	parts.append(wf.data.squeeze()) #squeeze brings shape from (16, 1, 512) -> (16, 512)
	# 	if count and len(parts) >= count:
	# 		break
	# return parts


def split_file_generator(filename, frequency_dimension=512, count = None, NUM_PARTS = 100):
	wf = Waterfall(filename, load_data = False)
	f_begin, f_end, f_delta = get_f_iter_bounds(wf.header, frequency_dimension)

	leftover = None
	#read NUM_PARTS from wf at a time, to be split after
	for iteration, (f_start, f_stop) in enumerate(pairwise(np.arange(f_begin, f_end, f_delta * NUM_PARTS))):
		wf.read_data(f_start=f_start, f_stop = f_stop)
		parts_together = wf.data.squeeze() #squeeze brings shape from (16, 1, 512 * NUM_PARTS) -> (16, 512 * NUM_PARTS)
		if not leftover is None:
			parts_together = np.append(leftover, parts_together, axis = 1)
		parts_lst, leftover = split_parts(parts_together, frequency_dimension)
		yield parts_lst

		if count and iteration * NUM_PARTS >= count:
			break
	if not count and not leftover is None:
		yield leftover


def split_parts(parts_together, frequency_dimension = 512):
	assert len(parts_together.shape) == 2 #should have been squeezed before calling this function
	parts = []
	i = 0
	total_length = parts_together.shape[1]
	while i + frequency_dimension <= total_length:
		part = parts_together[:, i:i+frequency_dimension]
		parts.append(part)
		i += frequency_dimension
	leftover = parts_together[:, i:] if i < total_length else None
	return parts, leftover



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
	parts = split_file(filename, count = 10000)
	write_parts_to_files(filename, parts)


if __name__ == "__main__":
	main()