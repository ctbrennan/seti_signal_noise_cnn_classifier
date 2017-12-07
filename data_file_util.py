import csv
from astropy.io import fits
import os
import numpy as np
from scipy import misc
import shutil

##############################################################################################################
############################################# Labeled Data ###################################################
##############################################################################################################
NUM_DISTINCT_LABELS = 2
label_dict = {'noise':0, 'broad':0, 'signa':1, 'lowsnr':-1}
labeled_file_dir = "./Combined"
label_filename = labeled_file_dir + "/labels.csv"
NUM_NOISE_TRAINING = 45459
NUM_SIGNAL_TRAINING = 4824
SN_DATA_RATIO = float(NUM_NOISE_TRAINING)/NUM_SIGNAL_TRAINING #change when we add more data

def prepare_training_data():
    """
    ARG1 - Directory name in which to find .fits files
    RET1 - array of normalized images (ndarrays)
    RET2 - array of ndarrays containing one-hot label vectors which also map image to a unique numeric id for use in dictionary d
    RET3 - dictionary which maps unique numeric id of image to (fullpath, class_label, fname)

    Note: bootstraps signal data so that RET1 has roughly as many signal images as noise images
    Note: assumes following directory structure with three levels below working directory i.e.: ./Combined/noise/HIP1324_1
    """
    print "Reading and training on files from {0}".format(labeled_file_dir)
    xs = []
    ys = []
    d = {}
    file_label_map = make_file_label_map()
    file_tuples = get_all_fits_info(labeled_file_dir, file_label_map)
    for unique_id, (fullpath, fname) in enumerate(file_tuples):
        x = get_and_normalize_data(fullpath)
        if x is None:
            continue
        class_label = file_label_map[fname]
        d[unique_id] = (fullpath, class_label, fname)
        y_vec = make_y_vec(class_label, unique_id)
        if class_label == 0:  
            xs.append(x)
            ys.append(y_vec)
        if class_label == 1:
            bootstrap_signal(xs, ys, x, y_vec)
    return np.array(xs), np.array(ys), d


def make_file_label_map(filename = label_filename):
    """
    ARGS - csv filename with labels
    RET - dictionary to map filename -> signal/noise label
    Side effects - 
    reads from csv file
    """
    file_label_map = {} #fname-> label
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            fname, label = row
            file_label_map[fname] = int(label)
    return file_label_map

def make_y_vec(class_label, unique_id):
    """
    ARG1 - integer class label
    ARG2 - unique numeric id for particular file
    RET - 2x2 matrix where first row is one-hot label vector, second row first column has unique id
    """
    y_vec = np.zeros((NUM_DISTINCT_LABELS, NUM_DISTINCT_LABELS))
    y_vec[0][class_label] = 1
    y_vec[1][0] = unique_id
    return y_vec

def bootstrap_signal(xs, ys, x, y_vec):
    """
    ARG1 - list of image ndarrays
    ARG2 - list of label ndarrays
    ARG3 - new image (labeled signal) to be added to ndarray some number of times according to a normal distribution
    ARG4 - new image label to be added the same number of times
    """
    std_dev = SN_DATA_RATIO/3.5 # set so that getting random value < 0 is very rare
    number_to_add = int(np.random.normal(SN_DATA_RATIO, std_dev))
    xs += number_to_add * [x]
    ys += number_to_add * [y_vec]

##############################################################################################################
########################################### Unlabeled Data ###################################################
##############################################################################################################
def prepare_unlabeled_data(root_file_dir):
    """
    ARG1 - Directory name in which to find .fits files
    RET1 - array of normalized images (ndarrays)
    RET2 - array of files from which those images came
    """
    xs = []
    filenames = []
    for subdir, dirs, files in os.walk(root_file_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            x = get_and_normalize_data(fullpath)
            if x is None:
                continue
            xs.append(x)
            filenames.append(fullpath)
    return np.array(xs), np.array(filenames)



##############################################################################################################
###########################################  Data/File Util ##################################################
##############################################################################################################

def get_and_normalize_data(fullpath):
    """
    ARG1 - String - Relative filename of .fits file of dimension (16, 512)
    RET - (ndarray - normalized image)
    Error RET - None

    """
    if not fullpath.endswith(".fits"):
        return None
    x = file_contents(fullpath)
    try:
        assert x.shape == (16,512)
    except AssertionError:
        print "File {0} is of size {1} instead of (16,512), so it won't be used".format(fullpath, x.shape)
        return None
    x = normalize_image(x) #trying whitening on single image level
    return x


def get_all_fits_info(directory, file_label_map, stop_short_count=None):
    """
    ARG1 - root directory in which to look for .fits files
    ARG2 (optional) - For testing purposes, a number of files to find so we can stop short
    
    RET - List of tuples of (fullpath, fname)
    Ex: If path of file found is ./Combined/noise/HIP1324_1/455634944.fits then
    fullpath = ./Combined/noise/HIP1324_1/455634944.fits 
    fname = HIP1324_1_455634944
    """
    file_tuples = []
    for subdir, dirs, files in os.walk(directory):
        dir_names = subdir.split("/")
        lowest_dir_name = dir_names[-1]
        for file in files:
            if stop_short_count and len(file_tuples) >= stop_short_count:
                break
            if not file.endswith(".fits"):
                continue
            filenumber = file[:-5]
            fname = lowest_dir_name + "_" + filenumber
            if fname in file_label_map:
                fullpath = os.path.join(subdir, file)
                file_tuples.append((fullpath, fname))
    return file_tuples

def file_contents(fname):
    """
    ARG1 - String - relative filename of .fits file
    RET - ndarray
    """
    hdulist = fits.open(fname, memmap=False)
    data = hdulist[0].data
    hdulist.close()
    return data

def normalize_image(image):
    """
    ARG1 - ndarray - 2d image
    RET -  de-meaned image scaled so that its pixel values have std. dev. of 1
    """
    avg = np.mean(image)
    stddev = np.std(image)
    image = (image-avg)/stddev
    return image

def copy_files_to_folder(files, directory, png = True):
    """
    ARG1 - Either: list of tuples where first index is filename, second is filename with directory or list of filenames without directory
    ex: [(46969856, GJ1002_1_46969856)] or [46969856]
    ARG2 - Directory to which we're copying files
    ARG3 - Boolean for whether we're copying existing png or existing .fits
    
    RET - None

    Side Effects - 
    deletes given directory
    copies .fits or .png with paths given in files to a given directory 
    """
    def change_extension_function_generator(old_extension = ".fits", new_extension = ".png"):
        if old_extension == new_extension:
            return lambda string: string  
        return lambda string: string[:len(string) - len(old_extension)] + new_extension

    if png:
        change_extension = change_extension_function_generator(".fits", ".png")
    else:
        change_extension = change_extension_function_generator(".fits", ".fits")
    

    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    files = [(change_extension(file), None) if type(file) != tuple else change_extension(file) for file in files]
    for file, fname_with_dir in files:
        try:
            shutil.copy2(file, directory)
            file_number = file.split("/")[-1]
            if fname_with_dir is not None:
                os.rename(directory + "/" + file_number, directory + "/" + fname_with_dir + ".png")
        except IOError:
            with open(no_image_found_file, "a") as f:
                f.write(file)

def find_all_fil(directory = "."):
    """
    ARG1 - Directory to look in
    RET - List of paths where filterbank files can be found
    """
    fils = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".fil"):
                fullpath = os.path.join(subdir, file)
                fils.append(fullpath)
    return fils


def recent_model_directory(directory):
    """
    ARG1 - Directory in which to look
    RET - Directory where the most recently modified file containing saved model was found
    """
    last_dir = None
    max_time = 0
    for subdir, dirs, files in os.walk(directory):
        if subdir == directory:
            continue
        modified_time = os.path.getmtime(subdir)
        if modified_time > max_time:
            last_dir = subdir
            max_time = modified_time
    return last_dir

def write_image_arrs_to_png(image_tups, directory, remake_dir = False):
    """
    ARG1 - list of image tuples of format (data array, filename)
    ARG2 - directory to which we write png
    ARG3 - (optional) boolean for whether we're deleting the given directory first
    RET - None
    Side effect - possibly deletes directory
    writes images in png format to given directory
    if "images" are of greater than 2 dimensions, treats ndarray as 2d images stacked on top of each other
    """
    if remake_dir or not os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)

    for data_arr, fname in image_tups:
        if len(data_arr.shape) == 1:
            data_arr = np.reshape(data_arr, (data_arr.shape[0], 1))
        if len(data_arr.shape) == 3:
            for dim in range(data_arr.shape[2]):
                write_image_arr_to_png(data_arr[:,:,dim], directory+"/"+fname+"-"+str(dim))
        else:
            write_image_arr_to_png(data_arr, directory + "/" + fname)

def write_image_arr_to_png(data_arr, fullpath):
    misc.imsave(fullpath, data_arr, format = "png")
