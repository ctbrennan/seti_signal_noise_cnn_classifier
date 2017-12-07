#!/usr/bin/env python
# coding: utf-8
import sys
import tensorflow as tf
from astropy.io import fits
import csv
import os
import tempfile
import numpy as np
import datetime
import shutil
from datetime import date
from generate_fits import split_file_generator
from scipy import misc

label_dict = {'noise':0, 'broad':0, 'signa':1, 'lowsnr':-1}
labeled_file_dir = "./Combined"
unlabeled_file_dir = "./generated_fits"
label_filename = labeled_file_dir + "/labels.csv"

no_image_found_file = "noCorrespondingPNG.txt"
testing_classification_dir = "./testing"
correctly_classified_dir = testing_classification_dir + "/correctly_classified"
misclassified_dir = testing_classification_dir + "/misclassified"
correct_noise_dir = correctly_classified_dir + "/noise"
correct_signal_dir = correctly_classified_dir + "/signal"
incorrect_noise_dir = misclassified_dir + "/noise"
incorrect_signal_dir = misclassified_dir + "/signal"

prediction_dir = "./predictions"
noise_prediction_dir = prediction_dir + "/noise"
signal_prediction_dir = prediction_dir + "/signal"

saved_model_dir = "./saved_models"

NUM_DISTINCT_LABELS = 2
NUM_NOISE_TRAINING = 45459
NUM_SIGNAL_TRAINING = 4824
SN_DATA_RATIO = float(NUM_NOISE_TRAINING)/NUM_SIGNAL_TRAINING #change when we add more data
BATCH_SIZE = 500
NUM_ITERATIONS = 1500#2000

def make_file_label_map():
    """
    ARGS - None
    RET - dictionary to map filename -> signal/noise label
    Side effects - 
    reads from csv file
    """
    file_label_map = {} #fname-> label
    with open(label_filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            fname, label = row
            file_label_map[fname] = int(label)
    return file_label_map

def file_contents(fname):
    """
    ARG1 - String - relative filename of .fits file
    RET - ndarray
    """
    hdulist = fits.open(fname, memmap=False)
    data = hdulist[0].data
    hdulist.close()
    return data

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

def normalize_image(image):
    """
    ARG1 - ndarray - 2d image
    RET -  de-meaned image scaled so that its pixel values have std. dev. of 1
    """
    avg = np.mean(image)
    stddev = np.std(image)
    image = (image-avg)/stddev
    return image

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

def prepare_training_data(root_file_dir):
    """
    ARG1 - Directory name in which to find .fits files
    RET1 - array of normalized images (ndarrays)
    RET2 - array of ndarrays containing one-hot label vectors which also map image to a unique numeric id for use in dictionary d
    RET3 - dictionary which maps unique numeric id of image to (fullpath, class_label, fname)

    Note: bootstraps signal data so that RET1 has roughly as many signal images as noise images
    Note: assumes following directory structure with three levels below working directory i.e.: ./Combined/noise/HIP1324_1
    """
    xs = []
    ys = []
    d = {}
    file_label_map = make_file_label_map()
    file_tuples = get_all_fits_info(root_file_dir, file_label_map)
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


def copy_files_to_folder(files, directory, png = True):
    """
    ARG1 - list of tuples where first index is filename, second is filename with directory 
    ex: [(46969856, GJ1002_1_46969856)]
    
    RET - None

    Side Effects - 
    deletes given directory
    copies .fits or .png with paths given in files to a given directory 
    """
    len_extension = len(".fits")
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)

    for file, fname_with_dir in files:
        if png:
            without_extension = file[:len(file) - len_extension]
            filename = without_extension + ".png"
        else:
            filename = file

        try:
            shutil.copy2(filename, directory)
            file_number = filename.split("/")[-1]
            os.rename(directory + "/" + file_number, directory + "/" + fname_with_dir + ".png")
        except IOError:
            with open(no_image_found_file, "a") as f:
                f.write(file)

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


def recent_model_directory(directory = saved_model_dir):
    """
    ARG1 - Directory in which to look
    RET - Directory where the most recently modified file containing saved model was found
    """
    last_dir = None
    max_time = 0
    for subdir, dirs, files in os.walk(directory):
        if subdir == saved_model_dir:
            continue
        modified_time = os.path.getmtime(subdir)
        if modified_time > max_time:
            last_dir = subdir
            max_time = modified_time
    return last_dir

def shuffle_in_unison(a, b):
    #https://stackoverflow.com/q/4601373
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b
    
def split_data_train_test(xs, ys):
    assert len(xs) == len(ys)
    TRAIN_PERCENT = .8 #test percent is 1-TRAIN_PERCENT
    cutoff_idx = int((TRAIN_PERCENT*len(xs))//1)
    xs, ys = shuffle_in_unison(xs, ys)
    train_x, test_x = xs[:cutoff_idx], xs[cutoff_idx:]
    train_y, test_y = ys[:cutoff_idx], ys[cutoff_idx:]
    training_data = [train_x, train_y]
    test_data = [test_x, test_y]
    return training_data, test_data

def conv2d_downsample_freq(x, W):
    # smaller stride along time dimension than frequency, try [1,1,2,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def build_cnn(x):
    """

    ARG1: a normalized input tensor with the dimensions (N_examples, 16, 512)

    RET1: tensor of shape (N_examples, 2)
    RET2: scalar placeholder for the probability of dropout
    """
    # pool by factor of two along frequency dimension -> bring dimension down to 128, max or average
    # more convolutional, either no fully connected or just later
    
    #check out batch normalization
    x_image = tf.reshape(x, [-1, 16, 512, 1])

    # First convolutional layer - maps image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d_downsample_freq(x_image, W_conv1) + b_conv1)
        h_conv1 = tf.identity(h_conv1, name="h_conv1")

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64, downsamples freq
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d_downsample_freq(h_pool1, W_conv2, ) + b_conv2)
        h_conv2 = tf.identity(h_conv2, name="h_conv2")

    # Second pooling layer.
    with tf.name_scope('pool2'):
        # h_pool2 = h_conv2
        h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer -- maps 64 feature maps to 128
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d_downsample_freq(h_pool2, W_conv3, ) + b_conv3)
        h_conv3 = tf.identity(h_conv3, name="h_conv3")

    # Third pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = h_conv3
        # h_pool3 = max_pool_2x2(h_conv3)


    num_pooling_layers = 2
    num_downsampling_conv_layers = 3    
    dim = 16 * 512 / 4**num_pooling_layers / 2**num_downsampling_conv_layers * 128


    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([dim, 128])
        b_fc1 = bias_variable([128])
        # maybe try smaller dimensionality

    h_pool3_flat = tf.reshape(h_pool3, [-1, dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1 = tf.identity(h_fc1, name = "h_fc1")

    # with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 128 features to 2 classes, one for each type of signal
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([128, 2])
        b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.identity(y_conv, name="y_conv")
    return y_conv, keep_prob

def build_graph(x, y_):

    y_conv, keep_prob = build_cnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        # separate into two steps, computing then applying gradient
        #put histogram summaries of length of gradient vector 
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)


    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return (accuracy, keep_prob, train_step, cross_entropy, merged_summary_op, correct_prediction)

def train_and_test(tensor_tup, x, y_, xs, ys, d):
    # accuracy, keep_prob, train_step, cross_entropy, merged_summary_op, correct_prediction = tensor_tup
    training_data, test_data = split_data_train_test(xs, ys)

    # saver = tf.train.Saver()
    dirname = saved_model_dir + "/" + str(datetime.datetime.now()) + "/"
    builder = tf.saved_model.builder.SavedModelBuilder(dirname)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess, training_data, tensor_tup, x, y_)
        misclassified_files, correctly_classified_files, full_accuracy = test(test_data, tensor_tup, x, y_, d)
        print('test accuracy %g' % full_accuracy)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()
    return misclassified_files, correctly_classified_files

def train(sess, training_data, tensor_tup, x, y_):
    accuracy, keep_prob, train_step, cross_entropy, merged_summary_op, correct_prediction = tensor_tup
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    batch_counter = batch_accuracy = 0
    last_5_batch_accuracies = [0 for _ in range(5)]
    for i in range(NUM_ITERATIONS):
        #stop it once it converges, shuffle training data after exhausting training batches
        if batch_counter >= len(training_data[0]):
            if batch_accuracy <= min(last_5_batch_accuracies):
                print (last_5_batch_accuracies)
                print ("breaking at step {}".format(i))
                break
            last_5_batch_accuracies.pop()
            last_5_batch_accuracies.append(batch_accuracy)
            batch_counter = batch_accuracy = 0
            training_data[0], training_data[1] = shuffle_in_unison(training_data[0], training_data[1])
        
        # Note: not normalizing on batch level because it doesn't seem to help, drops test acc to 49%
        x_batch = training_data[0][batch_counter: batch_counter + BATCH_SIZE, :, :]
        y_batch = training_data[1][batch_counter: batch_counter + BATCH_SIZE, 0, :]

        train_accuracy = accuracy.eval(feed_dict={
            x: x_batch, y_: y_batch, keep_prob: 1.0}, session=sess)
        batch_accuracy += train_accuracy
        _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op], 
                            feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
        if i % 10 == 0:
            writer.add_summary(summary, i)
            print('step %d, training accuracy %g' % (i, train_accuracy))
        batch_counter += BATCH_SIZE

def test(test_data, tensor_tup, x, y_, d):
    accuracy, keep_prob, train_step, cross_entropy, merged_summary_op, correct_prediction = tensor_tup
    batch_counter = accumulated_accuracy_sum = contributing_test_batches = 0
    orig_len = test_data[0].shape[0]
    mis_noise = set()
    mis_signal = set()
    corr_noise = set()
    corr_signal = set()
    
    while batch_counter < orig_len:
        x_test_batch = test_data[0][batch_counter: batch_counter + BATCH_SIZE, :, :]
        y_test_batch = test_data[1][batch_counter: batch_counter + BATCH_SIZE, 0, :]
        
        corr_pred = correct_prediction.eval(feed_dict={
            x: x_test_batch, y_: y_test_batch, keep_prob: 1.0})
        
        for idx, boolean in enumerate(tf.unstack(corr_pred)):
            unique_id = test_data[1][batch_counter + idx, 1, 0]
            fullpath, class_label, fname = d[unique_id]
            tup = (fullpath, fname)
            if (boolean.eval() != 1.0): # incorrect prediction
                if class_label == 0:
                    mis_noise.add(tup)
                else:
                    mis_signal.add(tup)    
            else:
                if class_label == 0:
                    corr_noise.add(tup)
                else:
                    # l = len(corr_signal)
                    corr_signal.add(tup)
                    # if l + 1 == len(corr_signal):
                        # normalized_img = 


        acc = tf.reduce_mean(corr_pred).eval()

        ratio = len(x_test_batch)/BATCH_SIZE
        accumulated_accuracy_sum += acc * ratio
        contributing_test_batches += 1 * ratio
        batch_counter += BATCH_SIZE
    full_accuracy = float(accumulated_accuracy_sum)/contributing_test_batches if contributing_test_batches != 0 else -1
    if full_accuracy == -1:
        print "No testing actually done, not enough data"
    misclassified_files = [mis_noise, mis_signal]
    correctly_classified_files = [corr_noise, corr_signal]
    return misclassified_files, correctly_classified_files, full_accuracy


def classify_all_fits_files(xs, filenames):
    dirname = recent_model_directory()
    noise = set()
    signal = set()
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], dirname)
        x = sess.graph.get_tensor_by_name("x:0")
        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
        y_conv = sess.graph.get_tensor_by_name("y_conv:0")
        i = 0
        while i < len(xs):
            batch = xs[i:i+BATCH_SIZE]
            filename_batch = filenames[i:i+BATCH_SIZE]
            pred = tf.argmax(y_conv.eval(feed_dict = {x: batch, keep_prob: 1.0}), 1).eval()
            for filename, class_label in zip(filename_batch, pred):
                if class_label == 0:
                    noise.add(filename)
                else:
                    signal.add(filename)
            i += BATCH_SIZE
       
    return noise, signal

def classify_lazily(write_signal = False):
    dirname = recent_model_directory()
    print "Most recent saved model coming from {0}".format(dirname)
    filenames = find_all_fil()
    noise = []
    signal = []
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], dirname)
        x = sess.graph.get_tensor_by_name("x:0")
        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
        y_conv = sess.graph.get_tensor_by_name("y_conv:0")
        h_conv1 = sess.graph.get_tensor_by_name("conv1/h_conv1:0")
        h_conv2 = sess.graph.get_tensor_by_name("conv2/h_conv2:0")
        h_conv3 = sess.graph.get_tensor_by_name("conv3/h_conv3:0")
        fc1 = sess.graph.get_tensor_by_name("h_fc1:0")

        
        for file in filenames:
            print "Reading lazily and classifying from {0}".format(file)
            file_number = 0
            for batch in split_file_generator(file, frequency_dimension=512, count = 10000, NUM_PARTS = 100):
                normalized_batch = [normalize_image(image_arr) for image_arr in batch]
                act1 = h_conv1.eval(feed_dict = {x: normalized_batch, keep_prob:1.0})
                act2 = h_conv2.eval(feed_dict = {x: normalized_batch, keep_prob:1.0})
                act3 = h_conv3.eval(feed_dict = {x: normalized_batch, keep_prob:1.0})
                act_fc = fc1.eval(feed_dict = {x: normalized_batch, keep_prob:1.0})
                pred = tf.argmax(y_conv.eval(feed_dict = {x: normalized_batch, keep_prob: 1.0}), 1).eval()
                for image_arr, class_label, act1, act2, act3, act_fc in zip(batch, pred, act1, act2, act3, act_fc):
                    part_filename = file + str(file_number)
                    image_tup = (image_arr, part_filename)
                    act_tup1 = (act1, part_filename + "-conv1")
                    act_tup2 = (act2, part_filename + "-conv2")
                    act_tup3 = (act3, part_filename + "-conv3")
                    act_tupfc = (act_fc, part_filename + "-fc1")
                    if class_label == 0:
                        noise.append(image_tup)
                    else:
                        signal.append(image_tup)
                        signal.append(act_tup1)
                        signal.append(act_tup2)
                        signal.append(act_tup3)
                        signal.append(act_tupfc)
                    file_number += 1
                if write_signal:
                    write_image_arrs_to_png(signal, signal_prediction_dir)
                    signal = []
       
    return noise, signal

def main():
    args = sys.argv
    if len(args) < 2:
        print "Specify mode with command line argument 'c' or 't'"
        return
    mode = args[1]
    if mode == 't':
        print "Reading and training on files from {0}".format(labeled_file_dir)
        xs, ys, d = prepare_training_data(labeled_file_dir)
        x = tf.placeholder(tf.float32, [None, 16, 512], name="x")
        y_ = tf.placeholder(tf.float32, [None, 2]) #noise/broad/lowsnr, signal
        tensor_tup = build_graph(x, y_)
        misclassified_files, correctly_classified_files = train_and_test(tensor_tup, x, y_, xs, ys, d)
        mis_noise, mis_signal = misclassified_files
        corr_noise, corr_signal = correctly_classified_files
        copy_files_to_folder(mis_noise, incorrect_noise_dir)
        copy_files_to_folder(mis_signal, incorrect_signal_dir)
        copy_files_to_folder(corr_noise, correct_noise_dir)
        copy_files_to_folder(corr_signal, correct_signal_dir)
    elif mode == 'cf':
        print "Reading and classifying files from {0}".format(unlabeled_file_dir)
        xs, filenames = prepare_unlabeled_data(unlabeled_file_dir)
        noise_filenames, signal_filenames = classify_all_fits_files(xs, filenames)
        copy_files_to_folder(noise_filenames, noise_prediction_dir, png = False)
        copy_files_to_folder(signal_filenames, signal_prediction_dir, png = False)
    elif mode == 'c':
        WRITE_SIGNAL_CONSTANTLY = True
        WRITE_NOISE = False

        noise_tups, signal_tups = classify_lazily(write_signal = WRITE_SIGNAL_CONSTANTLY)
        if not WRITE_SIGNAL_CONSTANTLY:
            print "Writing {0} predicted signal images to {1}".format(len(signal_tups), signal_prediction_dir)
            write_image_arrs_to_png(signal_tups, signal_prediction_dir)
        if WRITE_NOISE:
            print "Writing {0} predicted noise images to {1}".format(len(noise_tups), noise_prediction_dir)
            write_image_arrs_to_png(noise_tups, noise_prediction_dir)
        
    else:
        print "Mode not recognized"


if __name__ == "__main__":
    main()

'''
Todo:
work on visualization
given input, visualize activations of earlier layers
print precision, recall, confusion matrix
'''