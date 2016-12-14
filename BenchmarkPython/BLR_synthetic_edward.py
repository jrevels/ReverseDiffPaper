"""Bayesian linear regression using mean-field variational inference.

Modified from the original example of Edward: 
https://github.com/blei-lab/edward/blob/master/examples/bayesian_linear_regression.py

  Inputs
  ----------
  -n "number of iterations", default = 10000
  -bs "batchsize of MNIST dataset", default = 100
  -s "random seed of tensorflow", default = 12345

  Outputs
  ----------
  CSV file "BLR_lsclock.csv", contains n lines of benchmarking times for a single run of gradient in Tensorflow. Recommend to truncate the first X lines because it is a burn-in period for CPU. (Usually X >= 100)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import edward as ed
import numpy as np
import tensorflow as tf
import time
import argparse
import csv

from scipy.stats import norm
from edward.models import Normal, RandomVariable
from edward.util import get_session, get_variables, copy, kl_multivariate_normal
from memory_profiler import profile

# Take keyboard input to determine iter numbers and batchsize
parser = argparse.ArgumentParser(description="Timing the automatic differentiation time for Bayesian linear regression")
parser.add_argument("-n", "--n", type=int, help="number of iterations for AD", default=10000)
parser.add_argument("-samp", "--samp", type=int,  help="sample size for gradients; Note: the sample size doesn't affect the gradient time", default=100)
parser.add_argument("-s", "--seed", type=int,  help="tensorflow random seed", default=12345)

args = parser.parse_args()

print("\nBuilding model...")

ed.set_seed(args.seed)

# Function to generate synthetic data
def build_toy_dataset(N, noise_std=0.1):
    X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
    y = 5.0 * X + norm.rvs(0, noise_std, size=N)
    X = X.reshape((N, 1))
    return X.astype(np.float32), y.astype(np.float32)

# define a python memory profiler to print the memory allocation
@profile
def runsess(sess, var_grad, feed_dict):
    sess.run(var_grad, feed_dict=feed_dict)

# Model setup
N = 40  # number of data points
D = 1  # number of features

# DATA
X_train, y_train = build_toy_dataset(N)

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

data = {X: X_train, y: y_train}

# INFERENCE
qw = Normal(mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(mu=tf.Variable(tf.random_normal([1])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

latent_vars = {w: qw, b: qb}

# Initialize session
sess = get_session()
ph = tf.placeholder(tf.float32, y_train.shape)
var = tf.Variable(ph, trainable=False, collections=[])
sess.run(var.initializer, {ph: y_train})

# n_samples is the number of samples in building loss function
n_samples = args.samp
t = tf.Variable(0, trainable=False)
increment_t = t.assign_add(1)

# find the list of variables
var_list = set([])
trainables = tf.trainable_variables()

for z, qz in six.iteritems(latent_vars):
    if isinstance(z, RandomVariable):
        var_list.update(get_variables(z, collection=trainables))

    var_list.update(get_variables(qz, collection=trainables))

for x, qx in six.iteritems(data):
    if isinstance(x, RandomVariable) and not isinstance(qx, RandomVariable):
        var_list.update(get_variables(x, collection=trainables))

var_list = list(var_list)

# build a loss function
p_log_lik = [0.0] * n_samples

z_sample = {}
for z, qz in six.iteritems(latent_vars):
    # Copy q(z) to obtain new set of posterior samples.
    qz_copy = copy(qz)
    z_sample[z] = qz_copy.value()

dict_swap = z_sample
for x, qx in six.iteritems(data):
    if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
            qx_copy = copy(qx)
            dict_swap[x] = qx_copy.value()
        else:
            dict_swap[x] = qx

for x in six.iterkeys(data):
    if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap)
        x_log_lik = tf.reduce_sum(x_copy.log_prob(dict_swap[x]))

        p_log_lik[0] += x_log_lik

p_log_lik = tf.pack(p_log_lik)

kl = tf.reduce_sum([data.get(z, 1.0) * tf.reduce_sum(kl_multivariate_normal(
                        qz.mu, qz.sigma, z.mu, z.sigma))
                    for z, qz in six.iteritems(latent_vars)])

loss = -(tf.reduce_mean(p_log_lik) - kl)

# benchmark the gradient time
grads = tf.gradients(loss, [v.ref() for v in var_list])[0]

init = tf.initialize_all_variables()
feed_dict = {}
for key, value in six.iteritems(data):
    if isinstance(key, tf.Tensor):
        feed_dict[key] = value
init.run(feed_dict)

# run 10,000 iterations to benchmark the CPU time
lsclock = []
n_iter = args.n

print("Running gradients... (This may take a while)\n")

for _ in range(n_iter):
    start_clock = time.clock()
    var_grad_val = sess.run(grads, feed_dict)
    lsclock.append(time.clock() - start_clock)

# Printing out the memory allocation in Tensorflow graph:
# Once a certain memory is allocated for tensorflow graph, it seldom changes even in different iters.
# So we only need to print once.
# As the profiler is very slow, we separate the following function from above.
# 
print("Memory allocation:")
runsess(sess, grads, feed_dict)
# Note that the output MiB stands for "mebibyte" as 1MiB = 1048B

print("Outputing to csv file...")

with open("BLR_lsclock.csv","w") as filename:
    wr = csv.writer(filename, delimiter="\n")
    wr.writerow(lsclock)
