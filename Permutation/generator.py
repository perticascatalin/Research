import random
import pickle
import setup as stp

# Setup experiment size and parameters
N_CLASSES = stp.num_classes
N_SAMPLES = stp.num_samples
MAXINT = stp.maxint

# Additional number of features to generate alongside the data
# Order relation pairs
N_FEAT = (N_CLASSES*(N_CLASSES - 1))/2

def gen_list(dtype = 'int'):
	lst, order = list(), list()
	for i in range(N_CLASSES):
		while True:
			if dtype == 'float':
				num = random.random()
			else:
				num = random.randint(1, MAXINT)

			# Condition to generate unique numbers
			if lst.count(num) == 0:
				lst.append(num)
				break
			else:
				continue

	# Count number of elements smaller than each individual element
	# That number is its final position
	for i in range(N_CLASSES):
		count = 0
		for j in range(N_CLASSES):
			if lst[j] < lst[i]:
				count += 1
		order.append(count)
	return lst, order

# Just for int
def gen_ith(ith_target):
	lst, ith = list(), list()
	for i in range(N_CLASSES):
		while True:
			num = random.randint(1, MAXINT)

			# Condition to generate unique numbers
			if lst.count(num) == 0:
				lst.append(num)
				break
			else:
				continue
	# Find the ith element in the sorted list
	for i in range(N_CLASSES):
		count = 0
		for j in range(N_CLASSES):
			if lst[j] < lst[i]:
				count += 1
		if count == ith_target - 1:
			ith.append(i)
	return lst, ith

# Just numbers
def data(dtype = 'int'):
	lsts, orders = list(), list()
	for i in range(N_SAMPLES):
		lst, order = gen_list(dtype)
		lsts.append(lst)
		orders.append(order)
	return lsts, orders

# Minimum
def simple_data():
	lsts, orders = list(), list()
	for i in range(N_SAMPLES):
		lst, mins = gen_ith(1)
		lsts.append(lst)
		orders.append(mins)
	return lsts, orders

# Just order relations
def order_relations():
	lsts, orders = list(), list()
	for i in range(N_SAMPLES):
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

# Numbers and Order Relations
def all():
	lsts, orders = list(), list()
	for i in range(N_SAMPLES):
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