import os
import pickle
import tensorflow as tf
import numpy as np
import analysis as co
import generator as gen
import config as conf
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup experiment size and parameters
N_CLASSES = conf.num_inputs
batch_size = 128
data_type = "data"
lsts_train, orders_train = gen.data()

print "========================"
print "General Solution"

def neural_net(x, inputs, n_classes, num_labels, dropout, reuse, is_training):
	with tf.variable_scope('SortNet', reuse = reuse):
		v = tf.Variable(tf.zeros([1]), trainable = False)
	return None, None

lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

print X.shape

#logits_train, y_train = neural_net(X, Y, N_OUT_CLASSES, N_CLASSES, dropout, reuse = False, is_training = False)
ts = np.zeros((N_CLASSES, N_CLASSES * N_CLASSES), np.float32)
for i in range(N_CLASSES):
	for j in range(N_CLASSES):
		if i == j:
			continue

		ts[i][i*N_CLASSES + j] = 1.0
		ts[j][i*N_CLASSES + j] = -1.0

v = tf.Variable(ts, trainable = False)

print 'X shape:', X.shape
print 'V shape:', v.shape

R = tf.matmul(X, v)
Rf = tf.nn.sigmoid(1000 * R)
Rr = tf.reshape(Rf, [batch_size,N_CLASSES,N_CLASSES])
Rp = tf.reduce_sum(Rr, 2)
Rn = tf.cast(tf.add(-0.5, Rp), tf.int32)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

	sol, r, y, x = sess.run([Rn, R, Y, X])
	print 'Input'
	print x
	print 'Output'
	print y
	print 'Solution'
	print sol
	print 'Raw matrix'
	print r
	
print "========================"
print "Small Example"

n = 4
m = 2
x = np.zeros((m,n), np.float32)
x[0][0] = 21
x[0][1] = 7
x[0][2] = 16
x[0][3] = 3

x[1][0] = 12
x[1][1] = 11
x[1][2] = 6
x[1][3] = 14
# n*n = 16
y = np.zeros((n,n*n), np.float32)
for i in range(n):
	for j in range(n):
		if i == j:
			continue
		y[i][i*n + j] = 1.0
		y[j][i*n + j] = -1.0

X = tf.Variable(x, trainable = False)
Y = tf.Variable(y, trainable = False)
print 'X shape:', X
print 'Y shape:', Y

R = tf.matmul(X, Y)
print 'R shape:', R
Rf = tf.nn.sigmoid(1000 * R)
Rr = tf.reshape(Rf, [m,n,n])
Rp = tf.reduce_sum(Rr, 2)
Rn = tf.add(-0.5, Rp)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

	sol = sess.run([Rf])
	print len(sol)
	print len(sol[0])
	print len(sol[0][0])
	print sol[0][0]
	print sol[0][1]
	sol = np.reshape(sol, (m,n,n))
	print sol
	est = sess.run([Rn])
	print est

