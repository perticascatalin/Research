import os
import helper
import tensorflow as tf
import analysis as co
import generator as gen
import config as conf
import models as mod
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup experiment size and parameters
N_CLASSES     = conf.num_inputs
N_OUT_CLASSES = conf.num_outputs
data_type     = conf.data_type
task          = conf.task
form          = conf.form

learning_rate = 0.001
#num_steps = 100000
num_steps = 25000
display_step = 5000
batch_size = 64
model_name = "conv_rel_net_20"

checkpts_dir, stats_dir, results_dir, labels_dir = helper.make_dirs(model_name)
load_model = helper.load_model()

def tensor_conversion(lsts, mats, orders):
	lsts = tf.convert_to_tensor(lsts, dtype = tf.float32)
	mats = tf.expand_dims(tf.convert_to_tensor(mats, dtype = tf.float32), 3)
	orders = tf.convert_to_tensor(orders, dtype = tf.int32)
	lsts, mats, orders = tf.train.slice_input_producer([lsts, mats, orders], shuffle = True)
	return lsts, mats, orders

print "GENERATE TRAINING DATA"
lsts_train, mats_train, orders_train = gen.data_by_task_and_form(task, form, is_training = True)
lsts_train, mats_train, orders_train = tensor_conversion(lsts_train, mats_train, orders_train)

print "GENERATE VALIDATION DATA"
lsts_val, mats_val, orders_val = gen.data_by_task_and_form(task, form, is_training = False)
lsts_val, mats_val, orders_val = tensor_conversion(lsts_val, mats_val, orders_val)

X, Z, Y = tf.train.batch([lsts_train, mats_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
X_val, Z_val, Y_val = tf.train.batch([lsts_val, mats_val, orders_val], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)

logits_train = mod.conv_rel_net(Z,     N_OUT_CLASSES, N_CLASSES, reuse = False, is_training = True)
logits_test  = mod.conv_rel_net(Z,     N_OUT_CLASSES, N_CLASSES, reuse = True, is_training = False)
logits_valt  = mod.conv_rel_net(Z_val, N_OUT_CLASSES, N_CLASSES, reuse = True, is_training = True)
logits_val   = mod.conv_rel_net(Z_val, N_OUT_CLASSES, N_CLASSES, reuse = True, is_training = False)

train_loss_op = tf.constant(0.0, dtype = tf.float32)
val_loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	train_loss_op = train_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))
	val_loss_op = val_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_valt[i], labels = Y_val[:,i]))

correct_pred_train = tf.constant(0.0, dtype = tf.float32)
correct_pred_val = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUT_CLASSES):
	correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[i], 1), tf.cast(Y[:,i], tf.int64)), tf.float32)
	correct_pred_val = correct_pred_val + tf.cast(tf.equal(tf.argmax(logits_val[i], 1), tf.cast(Y_val[:,i], tf.int64)), tf.float32)
accuracy_train = tf.reduce_mean(correct_pred_train)
accuracy_val = tf.reduce_mean(correct_pred_val)

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(train_loss_op)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	if load_model:
		# Load model: https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		saver = tf.train.import_meta_graph(checkpts_dir + 'model.meta')
		saver.restore(sess, tf.train.latest_checkpoint(checkpts_dir))
		train_losses, val_losses, train_accs, val_accs, steps = co.loss_acc_load(stats_dir)
		start_step = steps[-1] * 1000 + 1
	else:
		# Run the initializer
		sess.run(init)
		train_losses, val_losses, train_accs, val_accs, steps = [], [], [], [], []
		start_step = 1

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

	# Training cycle
	for step in range(start_step, start_step + num_steps):
		sess.run(train_op)
		if step % display_step == 0:
			# Calculate average batch loss and accuracy
			training_loss, validation_loss, training_accuracy, validation_accuracy = 0.0, 0.0, 0.0, 0.0

			for i in range(100):
				train_loss, val_loss, acc_train, acc_val = sess.run([train_loss_op, val_loss_op, accuracy_train, accuracy_val])
				if i % 100 == 0:
					correct_pred, logits, y_exp, x = sess.run([correct_pred_val, logits_val, Y_val, X_val])
					co.debugger(correct_pred, logits, y_exp, x)
					co.print_pretty(correct_pred, logits, y_exp, x, step, labels_dir)
					# TODO: Update combine plots
				
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

	print("Optimization Finished!")
	# Dump additional data, plot it and save model
	co.loss_acc_dump(train_losses, val_losses, train_accs, val_accs, steps, stats_dir)
	co.plt_dump(train_losses, val_losses, train_accs, val_accs, steps, results_dir + '_sample.png')
	saver.save(sess, checkpts_dir + 'model')

	coord.request_stop()
	coord.join(threads)