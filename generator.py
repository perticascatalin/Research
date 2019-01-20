import random
import tensorflow as tf

LST_SIZE = 10

N_CLASSES = 10
dropout = 0.75
learning_rate = 0.001

def gen_list():
	lst = list()
	order = list()

	for i in range(LST_SIZE):
		lst.append(random.random())

	for i in range(LST_SIZE):
		count = 0
		for j in range(LST_SIZE):
			if lst[j] < lst[i]:
				count += 1
		order.append(count)

	return lst, order

def get_data():
	lsts, orders = list(), list()
	for i in range(128):
		lst, order = gen_list()
		lsts.append(lst)
		orders.append(order)

	lsts = tf.convert_to_tensor(lsts, dtype = tf.float32)
	orders = tf.convert_to_tensor(orders, dtype = tf.int32)

	return lsts, orders

lsts_train, orders_train = get_data()
lsts_val, orders_val = get_data()

X, Y = tf.train.batch([lsts_train, orders_train], batch_size = 16, capacity = 16 * 8, num_threads = 4)
X_val, Y_val = tf.train.batch([lsts_val, orders_val], batch_size = 16, capacity = 16 * 8, num_threads = 4)

def neural_net(x, n_classes, dropout, reuse, is_training):
	with tf.variable_scope('NeuralNet', reuse = reuse):
		fc1 = tf.layers.dense(x, 128)
		fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)

		outputs = list()
		for i in range(10):
			out_i = tf.layers.dense(fc1, n_classes)
			out_i = tf.nn.softmax(out_i) if not is_training else out_i
			outputs.append(out_i)

	return outputs

logits_train = neural_net(X, N_CLASSES, dropout, reuse = False, is_training = True)
logits_test = neural_net(X, N_CLASSES, dropout, reuse = True, is_training = False)
logits_val = neural_net(X_val, N_CLASSES, dropout, reuse = True, is_training = False)

loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(10):
	loss_op = loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))

print loss_op

# Define loss and optimizer (with train logits, for dropout to take effect)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

# a = tf.cast(tf.equal(tf.argmax(logits_val_1, 1), tf.cast(Y_val[:,0], tf.int64)), tf.int32)
# b = tf.cast(tf.equal(tf.argmax(logits_val_2, 1), tf.cast(Y_val[:,1], tf.int64)), tf.int32)
# c = tf.cast(tf.equal(tf.argmax(logits_val_3, 1), tf.cast(Y_val[:,2], tf.int64)), tf.int32)
# d = tf.cast(tf.equal(tf.argmax(logits_val_4, 1), tf.cast(Y_val[:,3], tf.int64)), tf.int32)
# f = tf.cast(tf.equal(tf.argmax(logits_val_5, 1), tf.cast(Y_val[:,4], tf.int64)), tf.int32)
# g = tf.cast(tf.equal(tf.argmax(logits_val_6, 1), tf.cast(Y_val[:,5], tf.int64)), tf.int32)
# h = tf.cast(tf.equal(tf.argmax(logits_val_7, 1), tf.cast(Y_val[:,6], tf.int64)), tf.int32)
# i = tf.cast(tf.equal(tf.argmax(logits_val_8, 1), tf.cast(Y_val[:,7], tf.int64)), tf.int32)
# j = tf.cast(tf.equal(tf.argmax(logits_val_9, 1), tf.cast(Y_val[:,8], tf.int64)), tf.int32)
# k = tf.cast(tf.equal(tf.argmax(logits_val_10, 1), tf.cast(Y_val[:,9], tf.int64)), tf.int32)

# #accuracy_val = tf.reduce_mean(tf.cast(correct_pred_val, tf.float32))
