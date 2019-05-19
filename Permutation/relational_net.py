import os
import pickle
import tensorflow as tf
import analysis as co
import generator as gen
import setup as stp

# Setup experiment size and parameters
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
N_CLASSES = stp.num_classes
N_OUT_CLASSES = stp.num_out_classes
N_FEAT = (N_CLASSES*(N_CLASSES - 1))/2
dropout = 0.0
learning_rate = 0.001
num_steps = 100000
display_step = 1000
batch_size = 128
layer_neurons = stp.layer_neurons
layer_dropout = stp.layer_dropout
num_layers = len(layer_neurons)
data_type = stp.data_type
model_name = "Q"

def relational_net(x, inputs, n_classes, num_labels, dropout, reuse, is_training):
	with tf.variable_scope('RelationalNet', reuse = reuse):

		units_1 = []
		for i in range(n_classes):
			for j in range(n_classes):
				# Combine 2 input units into a relational unit
				# This is pseudocode for now, but could be 2 possible ways of writing
				#rel_unit = tf.layers.dense([inputs[:,i], inputs[:,j]], 1)
				rel_unit = tf.layers.dense(tf.concat([[inputs[:,i]], [inputs[:,j]]], 1), 1, activation = tf.nn.tanh)
				units_1.append(rel_unit)

		units_2 = []
		for i in range(n_classes):
			agg_unit = tf.layers.dense(units_1[i*n_classes:(i+1)*n_classes], 1, activation = tf.nn.tanh)
			units_2.append(agg_unit)

		# These are the output units


		# # Define first layer
		# fc = tf.layers.dense(x, layer_neurons[0], activation = tf.nn.tanh)
		# if num_layers >= 2:
		# 	fc = tf.layers.dropout(fc, rate = layer_dropout[0], training = is_training)
		# layers = [fc]

		# # Define hidden layers
		# for i in range(1, num_layers - 1):
		# 	last_layer = layers[-1]
		# 	fc = tf.layers.dense(last_layer, layer_neurons[i], activation = tf.nn.tanh)
		# 	fc = tf.layers.dropout(fc, rate = layer_dropout[i], training = is_training)
		# 	layers.append(fc)

		# # Define last layer
		# if num_layers >= 2:
		# 	last_layer = layers[-1]
		# 	fc = tf.layers.dense(last_layer, layer_neurons[-1], activation = tf.nn.tanh)

		# # Define outputs
		# outputs = list()
		# for i in range(n_classes):
		# 	out_i = tf.layers.dense(fc, num_labels)
		# 	out_i = tf.nn.softmax(out_i) if not is_training else out_i
		# 	outputs.append(out_i)

	return outputs, inputs

if data_type == "data":
	print "DATA"
	lsts_train, orders_train = gen.data()
if data_type == "simple_data":
	print "SIMPLE DATA"
	lsts_train, orders_train = gen.simple_data()
elif data_type == "order_relations":
	print "ORDER RELATIONS"
	lsts_train, orders_train = gen.order_relations()
elif data_type == "all":
	print "ALL DATA"
	lsts_train, orders_train = gen.all()

print "TRAINING"

lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

if data_type == "data":
	lsts_val, orders_val = gen.data()
if data_type == "simple_data":
	lsts_val, orders_val = gen.simple_data()
elif data_type == "order_relations":
	lsts_val, orders_val = gen.order_relations()
elif data_type == "all":
	lsts_val, orders_val = gen.all()

print "VALIDATION"

lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)