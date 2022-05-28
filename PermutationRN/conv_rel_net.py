import os
import pickle
import pdb
import tensorflow as tf
import analysis as co
import generator as gen
import config as conf
import models as mod
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup experiment size and parameters
N_CLASSES = conf.num_inputs
N_OUT_CLASSES = conf.num_outputs
N_FEAT = (N_CLASSES*(N_CLASSES - 1))/2
learning_rate = 0.001
num_steps = 100000
display_step = 5000
batch_size = 64
layer_neurons = conf.layer_neurons
layer_dropout = conf.layer_dropout
num_layers = len(layer_neurons)
data_type = conf.data_type
model_name = "test_pre"

print "GENERATE TRAINING DATA"
lsts_train, orders_train = gen.data_by_type(data_type, is_training = True)
tf_lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
tf_orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)

pdb.set_trace()

init = tf.global_variables_initializer()
# saver = tf.train.Saver()
with tf.Session() as sess:
	# Run the initializer
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	pdb.set_trace()

# lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

# print "GENERATE VALIDATION DATA"
# lsts_val, orders_val = gen.data_by_type(data_type, is_training = False)
# lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
# orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
# lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)


