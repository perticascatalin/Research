import os
import pickle
import tensorflow as tf
import analysis as co
import generator as gen
import config as conf
import models as mod
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup experiment size and parameters
N_CLASSES = conf.num_inputs
N_OUT_CLASSES = conf.num_outputs
batch_size = 8
data_type = conf.data_type
model_name = "exp_8"

print "GENERATE TRAINING DATA"
lsts_train, orders_train = gen.data_by_type(data_type, is_training = True)
lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

print "GENERATE VALIDATION DATA"
lsts_val, orders_val = gen.data_by_type(data_type, is_training = False)
lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

units_a, units_b, units_c, units_d, units_e = mod.net_inspect(X,     N_OUT_CLASSES, N_CLASSES, batch_size, reuse = False, is_training = True)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	# Run the initializer
	sess.run(init)
	# Start the data queue
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

	Xa, Xb, Xc, Xd, Xe, Xinput = sess.run([units_a, units_b, units_c, units_d, units_e, X])
	pdb.set_trace()

	coord.request_stop()
	coord.join(threads)

