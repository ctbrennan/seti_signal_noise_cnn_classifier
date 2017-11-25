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

label_dict = {'noise':0, 'broad':0, 'signa':1, 'lowsnr':-1}
file_label_map = {} #fname-> label
labeled_file_dir = "./Combined"
unlabeled_file_dir = "./generated_fits"
label_filename = labeled_file_dir + "/labels.csv"
misclassified_dir = "./misclassified/"
no_image_found_file = "noCorrespondingPNG.txt"
saved_model_dir = "./saved_models/"
prediction_dir = "./predictions"
noise_prediction_dir = prediction_dir + "/noise"
signal_prediction_dir = prediction_dir + "/signal"

num_distinct_labels = 2
num_noise_training_samples = 45459
num_signal_training_samples = 4824
BATCH_SIZE = 500
NUM_ITERATIONS = 2000

def make_file_label_map():
    with open(label_filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            fname, label = row
            file_label_map[fname] = int(label)

def file_contents(fname):
    hdulist = fits.open(fname, memmap=False)
    data = hdulist[0].data
    hdulist.close()
    return data

def get_and_normalize_data(fullpath):
    if not fullpath.endswith(".fits"):
        return None
    x = file_contents(fullpath)
    try:
        assert x.shape == (16,512)
    except AssertionError:
        print "File {0} is of size {1} instead of (16,512), so it won't be used".format(fullpath, x.shape)
        return None
    x = normalize_images(x) #trying whitening on single image level
    return x

def normalize_images(images):
    avg = np.mean(images)
    stddev = np.std(images)
    images = (images-avg)/stddev
    return images

def prepare_unlabeled_data(root_file_dir):
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
    #assumes following directory structure with three levels below working directory i.e.: ./Combined/noise/HIP1324_1
    s_n_data_ratio = float(num_noise_training_samples)/num_signal_training_samples #change when we add more data
    std_dev = s_n_data_ratio/3.5 # set so that getting random value < 0 is very rare
    xs = []
    ys = []
    d = {}
    unique_id = 1
    for subdir, dirs, files in os.walk(root_file_dir):
        dir_names = subdir.split("/")
        if len(dir_names) < 4: # not in the level of directory where data is
            continue
        lowest_dir_name = dir_names[-1]
        for file in files:
            # if len(xs) >= 1000:
            #     break
            filenumber = file[:-5]
            fname = lowest_dir_name + "_" + filenumber
            if fname in file_label_map:
                fullpath = os.path.join(subdir, file) 
                x = get_and_normalize_data(fullpath)
                if x is None:
                    continue
                class_label = file_label_map[fname]
                y_vec = np.zeros((num_distinct_labels, num_distinct_labels))
                y_vec[0][class_label] = 1
                y_vec[1][0] = unique_id
                d[unique_id] = fullpath
                unique_id += 1
                number_to_add = 1 if class_label == 0 else int(np.random.normal(s_n_data_ratio, std_dev))
                if class_label == 0:  
                    xs.append(x)
                    ys.append(y_vec)
                if class_label == 1:
                    xs += number_to_add * [x]
                    ys += number_to_add * [y_vec]
    return np.array(xs), np.array(ys), d

def copy_files_to_folder(files, directory, png = True):
    len_extension = len(".fits")
    
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)

    for file in files:
        if png:
            without_extension = file[:len(file) - len_extension]
            filename = without_extension + ".png"
        else:
            filename = file

        try:
            shutil.copy2(filename, directory)
        except IOError:
            with open(no_image_found_file, "a") as f:
                f.write(file)

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

    ARG 1: a normalized input tensor with the dimensions (N_examples, 16, 512)

    RET 1: tensor of shape (N_examples, 2)
    RET 2: scalar placeholder for the probability of dropout
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

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64, downsamples freq
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d_downsample_freq(h_pool1, W_conv2, ) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        # h_pool2 = h_conv2
        h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer -- maps 64 feature maps to 128
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d_downsample_freq(h_pool2, W_conv3, ) + b_conv3)

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
    dirname = saved_model_dir + str(datetime.datetime.now()) + "/"
    builder = tf.saved_model.builder.SavedModelBuilder(dirname)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess, training_data, tensor_tup, x, y_)
        misclassified_files, full_accuracy = test(test_data, tensor_tup, x, y_, d)
        print('test accuracy %g' % full_accuracy)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()
    return files

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
    misclassified_files = set()
    while batch_counter < orig_len:
        x_test_batch = test_data[0][batch_counter: batch_counter + BATCH_SIZE, :, :]
        y_test_batch = test_data[1][batch_counter: batch_counter + BATCH_SIZE, 0, :]
        
        corr_pred = correct_prediction.eval(feed_dict={
            x: x_test_batch, y_: y_test_batch, keep_prob: 1.0})
        
        for idx, boolean in enumerate(tf.unstack(corr_pred)):
            if (boolean.eval() != 1.0): # incorrect prediction
                unique_id = test_data[1][batch_counter + idx, 1, 0]
                fullpath = d[unique_id]
                misclassified_files.add(fullpath)

        acc = tf.reduce_mean(corr_pred).eval()

        ratio = len(x_test_batch)/BATCH_SIZE
        accumulated_accuracy_sum += acc * ratio
        contributing_test_batches += 1 * ratio
        batch_counter += BATCH_SIZE
    full_accuracy = float(accumulated_accuracy_sum)/contributing_test_batches if contributing_test_batches != 0 else -1
    if full_accuracy == -1:
        print "No testing actually done, not enough data"
    return misclassified_files, full_accuracy


def classifyAll(xs, filenames):
    dirname = get_last_saved_dir()
    noise = set()
    signal = set()
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], dirname)
        x = sess.graph.get_tensor_by_name("x:0")
        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
        y_conv = sess.graph.get_tensor_by_name("y_conv:0")
        pred = tf.argmax(y_conv.eval(feed_dict = {x: xs, keep_prob: 1.0}), 1).eval()
        for filename, class_label in zip(filenames, pred):
            if class_label == 0:
                noise.add(filename)
            else:
                signal.add(filename)
    return noise, signal


def get_last_saved_dir():
    last_dir = None
    max_time = 0
    for subdir, dirs, files in os.walk(saved_model_dir):
        if subdir == saved_model_dir:
            continue
        modified_time = os.path.getmtime(subdir)
        if modified_time > max_time:
            last_dir = subdir
            max_time = modified_time
    return last_dir

def main():
    args = sys.argv
    if len(args) < 2:
        print "Specify mode with command line argument 'c' or 't'"
        return
    mode = args[1]
    if mode == 't':
        print "Reading and training on files from {0}".format(labeled_file_dir)
        make_file_label_map()
        xs, ys, d = prepare_training_data(labeled_file_dir)
        x = tf.placeholder(tf.float32, [None, 16, 512], name="x")
        y_ = tf.placeholder(tf.float32, [None, 2]) #noise/broad, signal, lowsnr
        tensor_tup = build_graph(x, y_)
        misclassified_files = train_and_test(tensor_tup, x, y_, xs, ys, d)
        copy_files_to_folder(misclassified_files, misclassified_dir)
    elif mode == 'c':
        print "Reading and classifying files from {0}".format(unlabeled_file_dir)
        xs, filenames = prepare_unlabeled_data(unlabeled_file_dir)
        noise, signal = classifyAll(xs, filenames)
        copy_files_to_folder(noise, noise_prediction_dir, png = False)
        copy_files_to_folder(signal, signal_prediction_dir, png = False)
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