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
batch_size = 64
layer_neurons = stp.layer_neurons
layer_dropout = stp.layer_dropout
num_layers = len(layer_neurons)
data_type = stp.data_type
model_name = "R"

def fully_relational_net(x, inputs, n_classes, num_labels, dropout, reuse, is_training):
	with tf.variable_scope('FullyRelationalNet', reuse = reuse):

		inputs = tf.cast(inputs, dtype = tf.float32)

		# Relational convolutional manner
		units_1 = []
		for i in range(n_classes):
			for j in range(n_classes):
				# Combine 2 input units into a relational unit
				a_unit = tf.slice(inputs, [0,i], [batch_size,1])
				b_unit = tf.slice(inputs, [0,j], [batch_size,1])
				rel_unit = tf.concat([a_unit, b_unit], 1)
				units_1.append(rel_unit)

		# Stack and create last dim channel [batch_sz, 2, NxN, 1]
		units_1a = tf.expand_dims(tf.stack(units_1, axis = 2), 3)

		# Aggregate with a convolution
		# num_channels: 1 (just one type of relation)
		# Returns [batch_sz, 1, NxN, 1]
		units_1b = tf.layers.conv2d(units_1a, 1, [2,1], [2,1], 'same')
		
		# Aggregate with a convolution
		# num_channels: 4 (random even small number)
		# filter, strides
		# same or valid "SAME will output the same input length, while VALID will not add zero padding"
		units_2 = tf.layers.conv2d(units_1b, 4, [1,n_classes], [1,n_classes], 'same')

		# Flatten: will result in [batch_sz,4*N] Tensor
		units_3 = tf.contrib.layers.flatten(units_2)

		# Define outputs: N softmaxes with N classes
		outputs = list()
		for i in range(n_classes):
			out_i = tf.layers.dense(units_3, num_labels)
			out_i = tf.nn.softmax(out_i) if not is_training else out_i
			outputs.append(out_i)
		# These are the output units
		# No dropout for now

	return outputs, inputs

lsts_train, orders_train = gen.data_by_type(data_type)
print "GENERATE TRAINING DATA"

lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

lsts_val, orders_val = gen.data_by_type(data_type)
print "GENERATE VALIDATION DATA"

lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

logits_train, y_train = fully_relational_net(X, Y, N_OUT_CLASSES, N_CLASSES, dropout, reuse = False, is_training = True)
logits_test, y_test = fully_relational_net(X, Y, N_OUT_CLASSES, N_CLASSES, dropout, reuse = True, is_training = False)
logits_val, y_val = fully_relational_net(X_val, Y_val, N_OUT_CLASSES, N_CLASSES, dropout, reuse = True, is_training = False)
logits_eye, y_eye = fully_relational_net(X_val, Y_val, N_OUT_CLASSES, N_CLASSES, dropout, reuse = True, is_training = False)

loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	loss_op = loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred_train = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[i], 1), tf.cast(Y[:,i], tf.int64)), tf.float32)
accuracy_train = tf.reduce_mean(correct_pred_train)

correct_pred_val = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	correct_pred_val = correct_pred_val + tf.cast(tf.equal(tf.argmax(logits_val[i], 1), tf.cast(Y_val[:,i], tf.int64)), tf.float32)
accuracy_val = tf.reduce_mean(correct_pred_val)

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
	pickle.dump(steps, open('./data/stats/' + model_name + '_ml_steps.p', 'wb'))
	# Or just plot it
	co.print_ltv(losses, train_accs, val_accs, steps, model_name + '_sample.png')
	# Save your model
	saver.save(sess, './checkpts/')
	# Stop threads
	coord.request_stop()
	coord.join(threads)