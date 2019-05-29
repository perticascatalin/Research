import os
import pickle
import tensorflow as tf
import analysis as co
import generator as gen
import setup as stp

# Setup experiment size and parameters
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Array of N inputs
N_CLASSES = stp.num_classes

# Array of N or other number of outputs
N_OUT_CLASSES = stp.num_out_classes

# Re-representation with relational features
N_FEAT = (N_CLASSES*(N_CLASSES - 1))/2

# General dropout, initially applied to all layers
dropout = 0.0

# Learning rate, inflences convergence of model (larger or smaller jumps in gradient descent)
learning_rate = 0.001

# The number of training steps
num_steps = 100000

# Displays loss, accuracy and sample classification every display_step iterations
display_step = 1000

# Number of samples per training step
batch_size = 128

# Array with number of neurons per layer
layer_neurons = stp.layer_neurons

# Array with dropout proportions from first layer to before last layer
layer_dropout = stp.layer_dropout

# Number of layers
num_layers = len(layer_neurons)

# Data re-representation
data_type = stp.data_type

# Model name for saving results
model_name = "a_10"

# Model names and their description
# D_20,24,28,30 -> [400,200] with comparisons and N = 20,24,28,30
# E_30 -> [1000,200] with comparisons and N = 30
# F_30,20 flat 1 layer -> [1000][400] and N = 30, 20
# G_30,20 flat 1 layer -> [30][20] and N = 30, 20

def neural_net(x, inputs, num_classes, num_labels, dropout, reuse, is_training):
	with tf.variable_scope('NeuralNet', reuse = reuse):
		# Comparison results by activation under baseline model (on simple data)
		# Sigmoid 6.6 
		# Relu X (not converge)
		# Tanh 8.8
		# Layers: first is input-dense with dropout, last is dense-classes no dropout

		# Define first layer
		fc = tf.layers.dense(x, layer_neurons[0], activation = tf.nn.tanh)
		if num_layers >= 2:
			fc = tf.layers.dropout(fc, rate = layer_dropout[0], training = is_training)
		layers = [fc]

		# Define hidden layers
		for i in range(1, num_layers - 1):
			last_layer = layers[-1]
			fc = tf.layers.dense(last_layer, layer_neurons[i], activation = tf.nn.tanh)
			fc = tf.layers.dropout(fc, rate = layer_dropout[i], training = is_training)
			layers.append(fc)

		# Define last layer
		if num_layers >= 2:
			last_layer = layers[-1]
			fc = tf.layers.dense(last_layer, layer_neurons[-1], activation = tf.nn.tanh)

		# Define outputs
		outputs = list()
		for i in range(num_classes):
			out_i = tf.layers.dense(fc, num_labels)
			out_i = tf.nn.softmax(out_i) if not is_training else out_i
			outputs.append(out_i)

	return outputs, inputs

# if data_type == "data":
# 	print "DATA"
# 	lsts_train, orders_train = gen.data()
# if data_type == "simple_data":
# 	print "SIMPLE DATA"
# 	lsts_train, orders_train = gen.simple_data()
# elif data_type == "order_relations":
# 	print "ORDER RELATIONS"
# 	lsts_train, orders_train = gen.order_relations()
# elif data_type == "all":
# 	print "ALL DATA"
# 	lsts_train, orders_train = gen.all_data()

lsts_train, orders_train = gen.data_by_type(data_type)
print "GENERATE TRAINING DATA"

lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

# if data_type == "data":
# 	lsts_val, orders_val = gen.data()
# if data_type == "simple_data":
# 	lsts_val, orders_val = gen.simple_data()
# elif data_type == "order_relations":
# 	lsts_val, orders_val = gen.order_relations()
# elif data_type == "all":
# 	lsts_val, orders_val = gen.all_data()

lsts_val, orders_val = gen.data_by_type(data_type)
print "GENERATE VALIDATION DATA"

lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

logits_train, y_train = neural_net(X, Y, N_OUT_CLASSES, N_CLASSES, dropout, reuse = False, is_training = True)
logits_test, y_test = neural_net(X, Y, N_OUT_CLASSES, N_CLASSES, dropout, reuse = True, is_training = False)
logits_val, y_val = neural_net(X_val, Y_val, N_OUT_CLASSES, N_CLASSES, dropout, reuse = True, is_training = False)
logits_eye, y_eye = neural_net(X_val, Y_val, N_OUT_CLASSES, N_CLASSES, dropout, reuse = True, is_training = False)

loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	loss_op = loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred_val = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	correct_pred_val = correct_pred_val + tf.cast(tf.equal(tf.argmax(logits_val[i], 1), tf.cast(Y_val[:,i], tf.int64)), tf.float32)
accuracy_val = tf.reduce_mean(correct_pred_val)

correct_pred_train = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[i], 1), tf.cast(Y[:,i], tf.int64)), tf.float32)
accuracy_train = tf.reduce_mean(correct_pred_train)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	# Run the initializer
	sess.run(init)
	# Start the data queue
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

	losses = []
	train_accs = []
	val_accs = []
	steps = []

	# Training cycle
	for step in range(1, num_steps+1):
		if step % display_step == 0:
			# Run optimization
			sess.run([train_op])
			# Calculate average batch loss and accuracy
			total_loss = 0.0
			training_accuracy = 0.0
			validation_accuracy = 0.0

			for i in range(100):
				loss, acc_train, acc_val = sess.run([loss_op, accuracy_train, accuracy_val])
				if i % 100 == 0:
					correct_pred, logits, y_exp, x = sess.run([correct_pred_val, logits_eye, Y_val, X_val])
					co.debugger(correct_pred, logits, y_exp, x)
					co.print_pretty(correct_pred, logits, y_exp, x, step)
				
				total_loss += loss
				training_accuracy += acc_train
				validation_accuracy += acc_val

			total_loss /= 100.0
			training_accuracy /= 100.0    
			validation_accuracy /= 100.0

			print("Step " + str(step) + ", Loss= " + \
				"{:.4f}".format(total_loss) + ", Training Accuracy= " + \
				"{:.3f}".format(training_accuracy) + ", Validation Accuracy= " + \
				"{:.3f}".format(validation_accuracy))

			losses.append(total_loss)
			train_accs.append(100.0*training_accuracy/N_CLASSES)
			val_accs.append(100.0*validation_accuracy/N_CLASSES)
			steps.append(step/1000)
		else:
			# Only run the optimization op (backprop)
			sess.run(train_op)

	print("Optimization Finished!")
	# Dump additional data for later investigation
	pickle.dump(losses, open('./data/stats/' + model_name + '_ml_losses.p', 'wb'))
	pickle.dump(train_accs, open('./data/stats/' + model_name + '_ml_t_accs.p', 'wb'))
	pickle.dump(val_accs, open('./data/stats/' + model_name + '_ml_v_accs.p', 'wb'))
	pickle.dump(val_accs, open('./data/stats/' + model_name + '_ml_steps.p', 'wb'))
	# Or just plot it
	co.print_ltv(losses, train_accs, val_accs, steps, model_name + '_sample.png')
	# Save your model
	saver.save(sess, './checkpts/')
	# Stop threads
	coord.request_stop()
	coord.join(threads)