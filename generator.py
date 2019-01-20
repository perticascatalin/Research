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

		out_1 = tf.layers.dense(fc1, n_classes)
		out_2 = tf.layers.dense(fc1, n_classes)
		out_3 = tf.layers.dense(fc1, n_classes)
		out_4 = tf.layers.dense(fc1, n_classes)
		out_5 = tf.layers.dense(fc1, n_classes)
		out_6 = tf.layers.dense(fc1, n_classes)
		out_7 = tf.layers.dense(fc1, n_classes)
		out_8 = tf.layers.dense(fc1, n_classes)
		out_9 = tf.layers.dense(fc1, n_classes)
		out_10 = tf.layers.dense(fc1, n_classes)

		out_1 = tf.nn.softmax(out_1) if not is_training else out_1
		out_2 = tf.nn.softmax(out_2) if not is_training else out_2
		out_3 = tf.nn.softmax(out_3) if not is_training else out_3
		out_4 = tf.nn.softmax(out_4) if not is_training else out_4
		out_5 = tf.nn.softmax(out_5) if not is_training else out_5
		out_6 = tf.nn.softmax(out_6) if not is_training else out_6
		out_7 = tf.nn.softmax(out_7) if not is_training else out_7
		out_8 = tf.nn.softmax(out_8) if not is_training else out_8
		out_9 = tf.nn.softmax(out_9) if not is_training else out_9
		out_10 = tf.nn.softmax(out_10) if not is_training else out_10

	return out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10

logits_train_1, logits_train_2, \
logits_train_3, logits_train_4, \
logits_train_5, logits_train_6, \
logits_train_7, logits_train_8, \
logits_train_9, logits_train_10 \
= neural_net(X, N_CLASSES, dropout, reuse = False, is_training = True)

logits_test_1, logits_test_2, \
logits_test_3, logits_test_4, \
logits_test_5, logits_test_6, \
logits_test_7, logits_test_8, \
logits_test_9, logits_test_10 \
= neural_net(X, N_CLASSES, dropout, reuse = True, is_training = False)

logits_val_1, logits_val_2, \
logits_val_3, logits_val_4, \
logits_val_5, logits_val_6, \
logits_val_7, logits_val_8, \
logits_val_9, logits_val_10 \
= neural_net(X_val, N_CLASSES, dropout, reuse = True, is_training = False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = \
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_1, labels=Y[:,0])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_2, labels=Y[:,1])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_3, labels=Y[:,2])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_4, labels=Y[:,3])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_5, labels=Y[:,4])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_6, labels=Y[:,5])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_7, labels=Y[:,6])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_8, labels=Y[:,7])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_9, labels=Y[:,8])) \
+ tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
logits=logits_train_10, labels=Y[:,9])) \

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

a = tf.cast(tf.equal(tf.argmax(logits_val_1, 1), tf.cast(Y_val[:,0], tf.int64)), tf.int32)
b = tf.cast(tf.equal(tf.argmax(logits_val_2, 1), tf.cast(Y_val[:,1], tf.int64)), tf.int32)
c = tf.cast(tf.equal(tf.argmax(logits_val_3, 1), tf.cast(Y_val[:,2], tf.int64)), tf.int32)
d = tf.cast(tf.equal(tf.argmax(logits_val_4, 1), tf.cast(Y_val[:,3], tf.int64)), tf.int32)
f = tf.cast(tf.equal(tf.argmax(logits_val_5, 1), tf.cast(Y_val[:,4], tf.int64)), tf.int32)
g = tf.cast(tf.equal(tf.argmax(logits_val_6, 1), tf.cast(Y_val[:,5], tf.int64)), tf.int32)
h = tf.cast(tf.equal(tf.argmax(logits_val_7, 1), tf.cast(Y_val[:,6], tf.int64)), tf.int32)
i = tf.cast(tf.equal(tf.argmax(logits_val_8, 1), tf.cast(Y_val[:,7], tf.int64)), tf.int32)
j = tf.cast(tf.equal(tf.argmax(logits_val_9, 1), tf.cast(Y_val[:,8], tf.int64)), tf.int32)
k = tf.cast(tf.equal(tf.argmax(logits_val_10, 1), tf.cast(Y_val[:,9], tf.int64)), tf.int32)

#accuracy_val = tf.reduce_mean(tf.cast(correct_pred_val, tf.float32))

