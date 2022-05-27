import random
import pickle
import config as conf

# Setup experiment size and parameters
N_CLASSES = conf.num_inputs
N_SAMPLES = conf.num_samples
MAXINT = conf.maxint

# Additional number of features to generate alongside the data
# Order relation pairs
N_FEAT = (N_CLASSES*(N_CLASSES - 1))/2

def gen_list(dtype = 'int'):
	lst, order = list(), list()
	used = [0] * (MAXINT + 1)
	for i in range(N_CLASSES):
		while True:
			if dtype == 'float':
				num = random.random()
				lst.append(num)
				break
			else:
				num = random.randint(1, MAXINT)
				# Condition to generate unique numbers
				if used[num] == False:
					lst.append(num)
					used[num] = True
					break

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
	used = [0] * (MAXINT + 1)
	for i in range(N_CLASSES):
		while True:
			num = random.randint(1, MAXINT)
			# Condition to generate unique numbers
			if used[num] == False:
				lst.append(num)
				used[num] = True
				break

	# Find the ith element in the sorted list
	for i in range(N_CLASSES):
		count = 0
		for j in range(N_CLASSES):
			if lst[j] < lst[i]:
				count += 1
		if count == ith_target - 1:
			ith.append(i)
	return lst, ith

# Three numbers to add (in range [0,1])
# Output will be rounded to nearest integer
def counter():
	lsts, orders = list(), list()
	for i in range(N_SAMPLES):
		x = random.random()
		y = random.random()
		z = random.random()
		lst = [x, y, z]
		order = [int(round(x + y + z))]
		lsts.append(lst)
		orders.append(order)
	return lsts, orders

# Two numbers to compare
def comparator():
	lsts, orders = list(), list()
	for i in range(N_SAMPLES):
		x = random.randint(1, MAXINT)
		y = random.randint(1, MAXINT)
		while y == x:
			y = random.randint(1, MAXINT)
		lst = [x, y]
		o = 1 if x > y else 0
		order = [o]
		lsts.append(lst)
		orders.append(order)
	return lsts, orders

# Longest increasing sequence
def gen_lis():
	lst, order = list(), list()
	for i in range(N_CLASSES):
		num = random.randint(1, MAXINT)
		max_seq = 0
		for j in range(len(lst)):
			if lst[j] < num and max_seq < order[j]:
				max_seq = order[j]
		lst.append(num)
		order.append(max_seq + 1)
	# subtract 1 to have numbers in range [0,N_CLASSES)
	for i in range(N_CLASSES):
		order[i] = order[i] - 1
	return lst, order

# Longest increasing sequence
def lis_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, order = gen_lis()
		lsts.append(lst)
		orders.append(order)
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, orders

# Minimum
def simple_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, mins = gen_ith(1)
		lsts.append(lst)
		orders.append(mins)
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, orders

# Just numbers
def data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, order = gen_list('int')
		lsts.append(lst)
		orders.append(order)
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, orders

# Just order relations
def order_relations(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
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
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, orders

# Numbers and Order Relations
def all_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
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
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, orders

# Get data by type
def data_by_type(data_type, is_training = True):
	n_samples = N_SAMPLES
	if not is_training:
		# Only generate 20% of samples for validation
		n_samples = int(n_samples / 5)

	if data_type == "lis":
		print ("LIS")
		return lis_data(n_samples)
	elif data_type == "simple_data":
		print ("SIMPLE DATA")
		return simple_data(n_samples)
	elif data_type == "data":
		print ("DATA")
		return data(n_samples)
	elif data_type == "order_relations":
		print ("ORDER RELATIONS")
		return order_relations(n_samples)
	elif data_type == "all":
		print ("ALL DATA")
		return all_data(n_samples)
