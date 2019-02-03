import os
import pickle
import tensorflow as tf
import analysis as co
import generator as gen

# Setup experiment size and parameters
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
N_CLASSES = 8
N_FEAT = (N_CLASSES*(N_CLASSES - 1))/2
dropout = 0.8
learning_rate = 0.001
num_steps = 100000
display_step = 1000
batch_size = 128

def neural_net(x, inputs, n_classes, dropout, reuse, is_training):
	with tf.variable_scope('NeuralNet', reuse = reuse):
		# activations tried: sigmoid 6.6 , relu X , tanh 8.8
		fc1 = tf.layers.dense(x, 516, activation = tf.nn.tanh)
		fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)
		fc2 = tf.layers.dense(fc1, 256, activation = tf.nn.tanh)
		fc2 = tf.layers.dropout(fc2, rate = dropout, training = is_training)
		fc3 = tf.layers.dense(fc2, 128, activation = tf.nn.tanh)
		outputs = list()
		for i in range(N_CLASSES):
			out_i = tf.layers.dense(fc3, n_classes)
			out_i = tf.nn.softmax(out_i) if not is_training else out_i
			outputs.append(out_i)
	return outputs, inputs

lsts_train, orders_train = gen.get_newer_data()
lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

lsts_val, orders_val = gen.get_newer_data()
lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

logits_train, y_train = neural_net(X, Y, N_CLASSES, dropout, reuse = False, is_training = True)
logits_test, y_test = neural_net(X, Y, N_CLASSES, dropout, reuse = True, is_training = False)
logits_val, y_val = neural_net(X_val, Y_val, N_CLASSES, dropout, reuse = True, is_training = False)
logits_eye, y_eye = neural_net(X_val, Y_val, N_CLASSES, dropout, reuse = True, is_training = False)

loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_CLASSES):
	loss_op = loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred_val = tf.constant(0.0, dtype = tf.float32)
for i in range(N_CLASSES):
	correct_pred_val = correct_pred_val + tf.cast(tf.equal(tf.argmax(logits_val[i], 1), tf.cast(Y_val[:,i], tf.int64)), tf.float32)
accuracy_val = tf.reduce_mean(correct_pred_val)

correct_pred_train = tf.constant(0.0, dtype = tf.float32)
for i in range(N_CLASSES):
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
					co.pretty_printing(correct_pred, logits, y_exp, x)
				#print acc_train
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
			train_accs.append(training_accuracy)
			val_accs.append(validation_accuracy)
		else:
			# Only run the optimization op (backprop)
			sess.run(train_op)

	print("Optimization Finished!")
	# Dump additional data for later investigation
	pickle.dump(losses, open('ml_losses.p', 'wb'))
	pickle.dump(train_accs, open('ml_train_accs.p', 'wb'))
	pickle.dump(val_accs, open('ml_val_accs.p', 'wb'))
	# Save your model
	saver.save(sess, './checkpts/')
	# Stop threads
	coord.request_stop()
	coord.join(threads)