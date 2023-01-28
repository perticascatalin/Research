import os
import pickle
import tensorflow as tf
import analysis as co
import generator as gen
import config as conf
import models as mod
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup experiment size and parameters
model_name    = "test"               # Model name for saving results
N_INPUTS      = conf.num_inputs      # Array of N inputs  (N_FEAT = (N_INPUTS*(N_INPUTS - 1))/2) if order relations
N_OUTPUTS     = conf.num_outputs     # Array of N outputs (or some other number)
data_type     = conf.data_type       # Data re-representation
task          = conf.task
form          = conf.form
layer_neurons = conf.layer_neurons   # Array with number of neurons per layer
layer_dropout = conf.layer_dropout   # Array with dropout proportions from first layer to before last layer
num_steps     = 100000               # The number of training steps
display_step  = 5000                 # Displays loss, accuracy and sample classification every display_step iterations
# display_step  = 100
# print_step    = 10000
batch_size    = 128                  # Number of samples per training step
learning_rate = 0.001                # Learning rate, inflences convergence of model (larger or smaller jumps in gradient descent)

checkpts_dir = './data/checkpts/' + model_name + '/'
stats_dir = './data/stats/' + model_name + '/'
results_dir = './data/results/' + model_name + '/'
labels_dir = results_dir + 'labels/'
dirs = [checkpts_dir, stats_dir, results_dir, labels_dir]
for direct in dirs:
	if not os.path.exists(direct):
		os.makedirs(direct)

print "GENERATE TRAINING DATA"
# lsts_train, orders_train = gen.data_by_type(data_type, is_training = True)
lsts_train, orders_train = gen.data_by_task_and_form(task, form, is_training = True)
lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)
lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)

print "GENERATE VALIDATION DATA"
# lsts_val, orders_val = gen.data_by_type(data_type, is_training = False)
lsts_val, orders_val = gen.data_by_task_and_form(task, form, is_training = False)
lsts_val = tf.convert_to_tensor(lsts_val, dtype = tf.float32)
orders_val = tf.convert_to_tensor(orders_val, dtype = tf.int32)
lsts_val, orders_val = tf.train.slice_input_producer([lsts_val, orders_val], shuffle = True)

# Organize data into batches
X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

# Define the logits for all datasets
logits_train = mod.neural_net(X,     N_INPUTS, N_OUTPUTS, layer_neurons, layer_dropout, reuse = False, is_training = True)
logits_test  = mod.neural_net(X,     N_INPUTS, N_OUTPUTS, layer_neurons, layer_dropout, reuse = True,  is_training = False)
logits_valt  = mod.neural_net(X_val, N_INPUTS, N_OUTPUTS, layer_neurons, layer_dropout, reuse = True,  is_training = True)
logits_val   = mod.neural_net(X_val, N_INPUTS, N_OUTPUTS, layer_neurons, layer_dropout, reuse = True,  is_training = False)

# Define the loss operation
train_loss_op = tf.constant(0.0, dtype = tf.float32)
val_loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUTPUTS):
	train_loss_op = train_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))
	val_loss_op = val_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_valt[i], labels = Y_val[:,i]))

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(train_loss_op)

# Define loss for prediction
correct_pred_train = tf.constant(0.0, dtype = tf.float32)
correct_pred_val = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUTPUTS):
	correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[i], 1), tf.cast(Y[:,i], tf.int64)), tf.float32)
	correct_pred_val   = correct_pred_val   + tf.cast(tf.equal(tf.argmax(logits_val[i], 1), tf.cast(Y_val[:,i], tf.int64)), tf.float32)
accuracy_train = tf.reduce_mean(correct_pred_train)
accuracy_val = tf.reduce_mean(correct_pred_val)	

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	# Run the initializer & Start the data queue
	# https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
	sess.run(init)
	# saver = tf.train.import_meta_graph(checkpts_dir + '.meta')
	# saver.restore(sess,tf.train.latest_checkpoint(checkpts_dir))

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	train_losses, val_losses, train_accs, val_accs, steps = [], [], [], [], []

	# Training cycle
	for step in range(1, num_steps+1):
		sess.run(train_op) # Run optimization
		if step % display_step == 0:
			# Calculate average batch loss and accuracy
			training_loss, validation_loss, training_accuracy, validation_accuracy = 0.0, 0.0, 0.0, 0.0

			# Each step walks through 100 x batch_size number of samples
			# Covers 12800/60000 = ~20% of dataset per interation
			for i in range(100):
				loss_train, loss_val, acc_train, acc_val = sess.run([train_loss_op, val_loss_op, accuracy_train, accuracy_val])
				# if i == 0 and step % print_step == 0:
				if i == 0:
					correct_pred, logits, y_exp, x = sess.run([correct_pred_val, logits_val, Y_val, X_val])
					co.debugger(correct_pred, logits, y_exp, x)
					co.print_pretty(correct_pred, logits, y_exp, x, step, labels_dir)
					# Also count strictly correctly sorted (uncomment next line)
					# co.print_pretty(correct_pred, logits, y_exp, x, step, labels_dir, True)
				
				training_loss += loss_train
				validation_loss += loss_val
				training_accuracy += acc_train
				validation_accuracy += acc_val

			training_loss /= 100.0
			validation_loss /= 100.0
			training_accuracy /= 100.0    
			validation_accuracy /= 100.0

			# if step % print_step == 0:
			print("Step " + str(step) + \
				", Training Loss= "       + "{:.4f}".format(training_loss) + \
				", Validation Loss= "     + "{:.4f}".format(validation_loss) + \
				", Training Accuracy= "   + "{:.3f}".format(training_accuracy) + \
				", Validation Accuracy= " + "{:.3f}".format(validation_accuracy))

			train_losses.append(training_loss)
			val_losses.append(validation_loss)
			train_accs.append(100.0*training_accuracy/N_INPUTS)
			val_accs.append(100.0*validation_accuracy/N_INPUTS)
			steps.append(step/1000)

	print("Optimization Finished!")

	# Dump additional data and plot it
	co.loss_acc_dump(train_losses, val_losses, train_accs, val_accs, steps, stats_dir)
	co.plt_dump(train_losses, val_losses, train_accs, val_accs, steps, results_dir + '_sample.png')

	# Save model
	saver.save(sess, checkpts_dir, global_step = 1000)

	# Stop threads
	coord.request_stop()
	coord.join(threads)