#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from astropy.io import fits
import csv
import os
import tempfile
import numpy as np
import datetime
import shutil

label_dict = {'noise':0, 'broad':0, 'signa':1, 'lowsnr':-1}
file_label_map = {} #fname-> label
root_file_dir = "./Combined"
label_filename = root_file_dir + "/labels.csv"
misclassified_dir = "./misclassified/"
no_image_found_file = "noCorrespondingPNG.txt"

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

def get_data(fname):
    hdulist = fits.open(fname, memmap=False)
    data = hdulist[0].data
    hdulist.close()
    return data

def normalize_images(images):
    avg = np.mean(images)
    stddev = np.std(images)
    images = (images-avg)/stddev
    return images

#bootstrap data
def get_data_from_files():
    s_n_data_ratio = float(num_noise_training_samples)/num_signal_training_samples #change when we add more data
    std_dev = s_n_data_ratio/3.5 # set so that getting random value < 0 is very rare
    num_distinct_labels = 2
    xs = []
    ys = []
    d = {}
    s = 0
    unique_id = 1
    for subdir, dirs, files in os.walk(root_file_dir):
        dir_names = subdir.split("/")
        if len(dir_names) >= 4:
            lowest_dir_name = dir_names[3]
        else:
            # not in the level of directory where data is
            continue
        for file in files:
            # if len(xs) >= 1000:
            #     break
            if not file.endswith(".fits"):
                continue
            fullpath = os.path.join(subdir, file)
            filenumber = file[:-5]
            fname = lowest_dir_name + "_" + filenumber
            if fname in file_label_map:
                x = get_data(fullpath)
                assert x.shape == (16,512)
                x = normalize_images(x) #trying whitening on single image level
                # x = x.reshape(16*512, 1)
                y = file_label_map[fname]
                y_vec = np.zeros((num_distinct_labels, num_distinct_labels))
                y_vec[0][y] = 1
                y_vec[1][0] = unique_id
                d[unique_id] = fullpath
                unique_id += 1
                number_to_add = 1 if y == 0 else int(np.random.normal(s_n_data_ratio, std_dev))
                if y == 0:  
                    xs.append(x)
                    ys.append(y_vec)
                if y == 1:
                    xs += number_to_add * [x]
                    ys += number_to_add * [y_vec]
    return np.array(xs), np.array(ys), d

def copy_files_to_folder(misclassified_set):
    len_extension = len(".fits")
    
    shutil.rmtree(misclassified_dir, ignore_errors=True)
    os.makedirs(misclassified_dir)

    for file in misclassified_set:
        without_extension = file[:len(file) - len_extension]
        png_filename = without_extension + ".png"
        try:
            shutil.copy2(png_filename, misclassified_dir)
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

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 128 features to 2 classes, one for each type of signal
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([128, 2])
        b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
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

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess, training_data, tensor_tup, x, y_)
        misclassified_set, full_accuracy = test(test_data, tensor_tup, x, y_, d)
        print('test accuracy %g' % full_accuracy)
        save_path = saver.save(sess, "./saved_models/" + str(full_accuracy) + "-" + str(datetime.datetime.now()))
    return misclassified_set

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
            x: x_batch, y_: y_batch, keep_prob: 1.0})
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
    misclassified_set = set()
    while batch_counter < orig_len:
        x_test_batch = test_data[0][batch_counter: batch_counter + BATCH_SIZE, :, :]
        y_test_batch = test_data[1][batch_counter: batch_counter + BATCH_SIZE, 0, :]
        
        corr_pred = correct_prediction.eval(feed_dict={
            x: x_test_batch, y_: y_test_batch, keep_prob: 1.0})
        
        for idx, boolean in enumerate(tf.unstack(corr_pred)):
            if (boolean.eval() != 1.0): # incorrect prediction
                unique_id = test_data[1][batch_counter + idx, 1, 0]
                fullpath = d[unique_id]
                misclassified_set.add(fullpath)

        acc = tf.reduce_mean(corr_pred).eval()

        ratio = len(x_test_batch)/BATCH_SIZE
        accumulated_accuracy_sum += acc * ratio
        contributing_test_batches += 1 * ratio
        batch_counter += BATCH_SIZE
    full_accuracy = float(accumulated_accuracy_sum)/contributing_test_batches if contributing_test_batches != 0 else -1
    return misclassified_set, full_accuracy

def main():
    make_file_label_map()
    xs, ys, d = get_data_from_files()

    x = tf.placeholder(tf.float32, [None, 16, 512])
    y_ = tf.placeholder(tf.float32, [None, 2]) #noise/broad, signal, lowsnr
    tensor_tup = build_graph(x, y_)
    misclassified_set = train_and_test(tensor_tup, x, y_, xs, ys, d)

    copy_files_to_folder(misclassified_set)

if __name__ == "__main__":
    main()

'''
Todo:
work on visualization
given input, visualize activations of earlier layers
print precision, recall, confusion matrix
'''