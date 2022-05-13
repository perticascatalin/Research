import os
import pickle
import tensorflow as tf
import analysis as co
import generator as gen
import setup as stp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup experiment size and parameters
model_name    = "test"              # Model name for saving results
N_LABELS      = stp.num_labels      # Array of N inputs (N_FEAT = (N_LABELS*(N_LABELS - 1))/2) if order relations
N_CLASSES     = stp.num_out_classes # Array of N or other number of outputs
data_type     = stp.data_type       # Data re-representation
layer_neurons = stp.layer_neurons   # Array with number of neurons per layer
layer_dropout = stp.layer_dropout   # Array with dropouts for all layers but last
num_layers    = len(layer_neurons)  # Number of layers
learning_rate = 0.001               # Learning rate, inflences convergence of model (larger or smaller jumps in gradient descent)
num_steps     = 100000              # The number of training steps
display_step  = 5000                # Displays loss, accuracy and sample classification every display_step iterations
batch_size    = 128                 # Number of samples per training step

def neural_net(x, num_classes, num_labels, reuse, is_training):
	with tf.variable_scope('NeuralNet', reuse = reuse):
		# Comparative results by activation under baseline model (on simple data)
		# Sigmoid: 6.6, Tanh: 8.8, Relu: X (no convergence)
		# Layers: first is input-dense with dropout, last is dense-classes no dropout

		layers = []
		for i in range(0, num_layers):
			last_layer = x if i == 0 else layers[-1]
			this_layer = tf.layers.dense(last_layer, layer_neurons[i], activation = tf.nn.tanh)
			this_layer = this_layer if i == num_layers - 1 else tf.layers.dropout(this_layer, rate = layer_dropout[i], training = is_training)
			layers.append(this_layer)

		# Define outputs (note: num_labels can differ by i)
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(this_layer, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)

	return outputs

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

# Organize data into batches
X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

# Define the logits for all datasets
logits_train = neural_net(X,     N_CLASSES, N_LABELS, reuse = False, is_training = True)
logits_test  = neural_net(X,     N_CLASSES, N_LABELS, reuse = True,  is_training = False)
logits_valt  = neural_net(X_val, N_CLASSES, N_LABELS, reuse = True,  is_training = True)
logits_val   = neural_net(X_val, N_CLASSES, N_LABELS, reuse = True,  is_training = False)

# Define the loss operation
train_loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_CLASSES):
	train_loss_op = train_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))

val_loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_CLASSES):
	val_loss_op = val_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_valt[i], labels = Y_val[:,i]))

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(train_loss_op)

# Define loss for prediction on training dataset
correct_pred_train = tf.constant(0.0, dtype = tf.float32)
for i in range(N_CLASSES):
	correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[i], 1), tf.cast(Y[:,i], tf.int64)), tf.float32)
accuracy_train = tf.reduce_mean(correct_pred_train)

# Define loss for prediction on validation dataset
correct_pred_val = tf.constant(0.0, dtype = tf.float32)
for i in range(N_CLASSES):
	correct_pred_val = correct_pred_val + tf.cast(tf.equal(tf.argmax(logits_val[i], 1), tf.cast(Y_val[:,i], tf.int64)), tf.float32)
accuracy_val = tf.reduce_mean(correct_pred_val)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	# Run the initializer & Start the data queue
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

	train_losses = []
	val_losses = []
	train_accs = []
	val_accs = []
	steps = []

	# Training cycle
	for step in range(1, num_steps+1):
		if step % display_step == 0:
			# Run optimization
			sess.run([train_op])
			# Calculate average batch loss and accuracy
			training_loss = 0.0
			validation_loss = 0.0
			training_accuracy = 0.0
			validation_accuracy = 0.0

			# Each step walks through 100 x batch_size number of samples
			# Covers 12800/60000 = ~20% of dataset in an interation
			for i in range(100):
				loss_train, loss_val, acc_train, acc_val = sess.run([train_loss_op, val_loss_op, accuracy_train, accuracy_val])
				if i == 0:
					correct_pred, logits, y_exp, x = sess.run([correct_pred_val, logits_val, Y_val, X_val])
					co.debugger(correct_pred, logits, y_exp, x)
					co.print_pretty(correct_pred, logits, y_exp, x, step)
					# also count strictly correctly sorted (uncomment next line)
					#co.print_pretty(correct_pred, logits, y_exp, x, step, True)
				
				training_loss += loss_train
				validation_loss += loss_val
				training_accuracy += acc_train
				validation_accuracy += acc_val

			training_loss /= 100.0
			validation_loss /= 100.0
			training_accuracy /= 100.0    
			validation_accuracy /= 100.0

			print("Step " + str(step) + \
				", Training Loss= "       + "{:.4f}".format(training_loss) + \
				", Validation Loss= "     + "{:.4f}".format(validation_loss) + \
				", Training Accuracy= "   + "{:.3f}".format(training_accuracy) + \
				", Validation Accuracy= " + "{:.3f}".format(validation_accuracy))

			train_losses.append(training_loss)
			val_losses.append(validation_loss)
			train_accs.append(100.0*training_accuracy/N_LABELS)
			val_accs.append(100.0*validation_accuracy/N_LABELS)
			steps.append(step/1000)
		else:
			# Only run the optimization op (backprop)
			sess.run(train_op)

	print("Optimization Finished!")
	# Dump additional data for later investigation
	pickle.dump(train_losses, open('./data/stats/' + model_name + '_ml_t_losses.p', 'wb'))
	pickle.dump(val_losses, open('./data/stats/' + model_name + '_ml_v_losses.p', 'wb'))
	pickle.dump(train_accs, open('./data/stats/' + model_name + '_ml_t_accs.p', 'wb'))
	pickle.dump(val_accs, open('./data/stats/' + model_name + '_ml_v_accs.p', 'wb'))
	pickle.dump(steps, open('./data/stats/' + model_name + '_ml_steps.p', 'wb'))

	# Plot data and save model
	co.print_ltv(train_losses, val_losses, train_accs, val_accs, steps, model_name + '_sample.png')
	saver.save(sess, './checkpts/')

	# Stop threads
	coord.request_stop()
	coord.join(threads)