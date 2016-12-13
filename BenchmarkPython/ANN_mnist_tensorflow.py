"""A very simple MNIST classifier.

Modified from the original example from Tensorflow website: https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/mnist/mnist_softmax.py

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

  Inputs
  ----------
  -n "number of iterations", default = 10000
  -bs "batchsize of MNIST dataset", default = 100
  -s "random seed of tensorflow", default = 12345


  Outputs
  ----------
  CSV file "lsclock.csv", contains n lines of benchmarking times for a single run of gradient in Tensorflow. Recommend to truncate the first X lines because it is a burn-in period for CPU. (Usually X=200)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import argparse
import csv

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

# Take keyboard input to determine iter numbers and batchsize
parser = argparse.ArgumentParser(description="Timing the automatic differentiation time in Tensorflow")
parser.add_argument("-n", "--n", type=int, help="number of iterations for AD", default=10000)
parser.add_argument("-bs", "--bs", type=int,  help="batchsize for MNIST training set", default=100)
parser.add_argument("-s", "--seed", type=int,  help="tensorflow random seed", default=12345)

args = parser.parse_args()

# load in MNIST training data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Store timing in lsclock 
lsclock = []

# Note: larger batch size can result in longer gradient time
# Mnist has 60,000 files in the training dataset
bs = args.bs

# open a new session every time
with tf.Session() as sess:
    tf.set_random_seed(args.seed)
    #tf.initialize_all_variables().run()
    sess.run(tf.initialize_all_variables())
    
    y_ = tf.placeholder(tf.float32, [None, 10])
    # cost function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # gradient virtual node
    var_grad = tf.gradients(cross_entropy, [W, b])[0]
    
    for i in range(args.n):
        # get new batch
        batch_xs, batch_ys = mnist.train.next_batch(bs)
        # set start clock
        start_clock = time.clock()
        # this is the step to run gradient with actual batch
        var_grad_val = sess.run(var_grad, feed_dict={x: batch_xs, y_: batch_ys})
        lsclock.append(time.clock() - start_clock)

with open("lsclock.csv","w") as filename:
    wr = csv.writer(filename, delimiter="\n")
    wr.writerow(lsclock)
