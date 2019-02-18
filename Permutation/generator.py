import random
import pickle

# Setup experiment size and parameters
N_CLASSES = 10
N_FEAT = (N_CLASSES*(N_CLASSES - 1))/2
MAXINT = 50

def gen_list(dtype = 'int'):
	lst, order = list(), list()
	for i in range(N_CLASSES):
		while True:
			if dtype == 'float':
				num = random.random()
			else:
				num = random.randint(1, MAXINT)

			if lst.count(num) == 0:
				lst.append(num)
				break
			else:
				continue
	for i in range(N_CLASSES):
		count = 0
		for j in range(N_CLASSES):
			if lst[j] < lst[i]:
				count += 1
		order.append(count)
	return lst, order

# just numbers
def get_data(dtype = 'int'):
	lsts, orders = list(), list()
	for i in range(32768):
		lst, order = gen_list(dtype)
		lsts.append(lst)
		orders.append(order)
	return lsts, orders

# just order relations
def get_new_data():
	lsts, orders = list(), list()
	for i in range(32768):
		lst, order = gen_list('int')
		c_lst = list()
		for j in range(N_CLASSES - 1):
			for k in range(j+1, N_CLASSES):
				if lst[j] > lst[k]:
					c_lst.append(1)
				else:
					c_lst.append(0)
		lsts.append(c_lst)
		orders.append(order)
	return lsts, orders

# numbers and order relations
def get_newer_data():
	lsts, orders = list(), list()
	for i in range(32768):
		lst, order = gen_list('int')
		c_lst = lst
		for j in range(N_CLASSES - 1):
			for k in range(j+1, N_CLASSES):
				if lst[j] > lst[k]:
					c_lst.append(1)
				else:
					c_lst.append(0)
		lsts.append(c_lst)
		orders.append(order)
	return lsts, orders