import os
import pickle
import numpy as np
import tensorflow as tf
import analysis as co
import generator as gen
import setup as stp

# Setup experiment size and parameters
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
learning_rate = 0.001
num_steps = 4000
display_step = 10
batch_size = 64
model_name = "ccb"

def cuboid_counter(x, inputs, reuse, is_training):
	with tf.variable_scope('CuboidCounter', reuse = reuse):
		# Define perceptron
		#perceptron = tf.layers.dense(x, 1, activation = tf.nn.sigmoid)

		# Try with more units to check if maybe not learnable with 1 unit
		perceptron = tf.layers.dense(x, 4, activation = tf.nn.tanh)

		# Try with more layers if maybe not learnable with 1 layer
		perceptron = tf.layers.dense(perceptron, 8, activation = tf.nn.tanh)
		
		# Define output
		output = tf.layers.dense(perceptron, 4)
		output = tf.nn.softmax(output) if not is_training else output

	return [output], inputs

# Get training data
lsts_train, orders_train = gen.counter()

lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

# Get validation data
lsts_val, orders_val = gen.counter()

lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

logits_train, y_train = cuboid_counter(X, Y, reuse = False, is_training = True)
logits_test, y_test = cuboid_counter(X, Y, reuse = True, is_training = False)
logits_val, y_val = cuboid_counter(X_val, Y_val, reuse = True, is_training = False)


loss_op = tf.constant(0.0, dtype = tf.float32)
loss_op = loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[0], labels = Y[:,0]))

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred_val = tf.constant(0.0, dtype = tf.float32)
correct_pred_val = correct_pred_val + tf.cast(tf.equal(tf.argmax(logits_val[0], 1), tf.cast(Y_val[:,0], tf.int64)), tf.float32)
accuracy_val = tf.reduce_mean(correct_pred_val)

correct_pred_train = tf.constant(0.0, dtype = tf.float32)
correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[0], 1), tf.cast(Y[:,0], tf.int64)), tf.float32)
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

	cp, logs, yexp, xs = [], [], [], []
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
					correct_pred, logits, y_exp, x = sess.run([correct_pred_val, logits_val, Y_val, X_val])
					cp.extend(correct_pred)
					#logs.extend(logits)
					if len(logs) == 0:
						logs = np.squeeze(np.array(logits), axis = 0)
					else:
						print logs.shape
						logits = np.squeeze(np.array(logits), axis = 0)
						print logits.shape
						logs = np.append(logs, logits, axis = 0)
					yexp.extend(y_exp)
					xs.extend(x)

				total_loss += loss
				training_accuracy += acc_train
				validation_accuracy += acc_val

			if step % 100 == 0:
				co.debugger_whole_batch_cuboid(cp, logs, yexp, xs, step)
				cp, logs, yexp, xs = [], [], [], []

			total_loss /= 100.0
			training_accuracy /= 100.0    
			validation_accuracy /= 100.0

			print("Step " + str(step) + ", Loss= " + \
				"{:.4f}".format(total_loss) + ", Training Accuracy= " + \
				"{:.3f}".format(training_accuracy) + ", Validation Accuracy= " + \
				"{:.3f}".format(validation_accuracy))

			losses.append(total_loss)
			train_accs.append(100.0*training_accuracy)
			val_accs.append(100.0*validation_accuracy)
			steps.append(step/100)
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
