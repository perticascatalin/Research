import random
import tensorflow as tf

LST_SIZE = 10

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

lsts, orders = list(), list()
for i in range(128):
	lst, order = gen_list()
	lsts.append(lst)
	orders.append(order)

lsts = tf.convert_to_tensor(lsts, dtype = tf.float32)
orders = tf.convert_to_tensor(orders, dtype = tf.int32)

X, Y = tf.train.batch([lsts, orders], batch_size = 16, capacity = 16 * 8, num_threads = 4)
print X, Y
# 30
# 30