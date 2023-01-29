import os
import pickle
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
model_name = "retest"

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

logits_train = mod.norm_conv_rel_net(X,     N_OUT_CLASSES, N_CLASSES, batch_size, reuse = False, is_training = True)
logits_test  = mod.norm_conv_rel_net(X,     N_OUT_CLASSES, N_CLASSES, batch_size, reuse = True, is_training = False)
logits_valt  = mod.norm_conv_rel_net(X_val, N_OUT_CLASSES, N_CLASSES, batch_size, reuse = True, is_training = True)
logits_val   = mod.norm_conv_rel_net(X_val, N_OUT_CLASSES, N_CLASSES, batch_size, reuse = True, is_training = False)

train_loss_op = tf.constant(0.0, dtype = tf.float32)
val_loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	train_loss_op = train_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))
	val_loss_op = val_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_valt[i], labels = Y_val[:,i]))

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(train_loss_op)

correct_pred_train = tf.constant(0.0, dtype = tf.float32)
correct_pred_val = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[i], 1), tf.cast(Y[:,i], tf.int64)), tf.float32)
	correct_pred_val = correct_pred_val + tf.cast(tf.equal(tf.argmax(logits_val[i], 1), tf.cast(Y_val[:,i], tf.int64)), tf.float32)
accuracy_train = tf.reduce_mean(correct_pred_train)
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

			for i in range(100):
				train_loss, val_loss, acc_train, acc_val = sess.run([train_loss_op, val_loss_op, accuracy_train, accuracy_val])
				if i % 100 == 0:
					correct_pred, logits, y_exp, x = sess.run([correct_pred_val, logits_val, Y_val, X_val])
					co.debugger(correct_pred, logits, y_exp, x)
					co.print_pretty(correct_pred, logits, y_exp, x, step)
				
				training_loss += train_loss
				validation_loss += val_loss
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
			train_accs.append(100.0*training_accuracy/N_CLASSES)
			val_accs.append(100.0*validation_accuracy/N_CLASSES)
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
	co.plt_dump(train_losses, val_losses, train_accs, val_accs, steps, model_name + '_sample.png')
	saver.save(sess, './checkpts/')

	# Stop threads
	coord.request_stop()
	coord.join(threads)