import random
import pickle
import pdb
import config as conf

# Setup experiment size and parameters
N_CLASSES = conf.num_inputs
N_SAMPLES = conf.num_samples
MAXINT = conf.maxint

# Generate a list
def gen_list():
	lst = []
	used = [0] * (MAXINT + 1)
	for i in range(N_CLASSES):
		while True:
			num = random.randint(1, MAXINT)
			if used[num] == False:
				lst.append(num)
				used[num] = True
				break
	return lst

# Get output for sort task
def get_sort(lst):
	res = []
	for i in range(N_CLASSES):
		count = 0
		for j in range(N_CLASSES):
			if lst[j] < lst[i]:
				count += 1
		res.append(count)
	return res

# Get output for lis task
def get_lis(lst):
	res = []
	for i in range(N_CLASSES):
		num = lst[i]
		max_seq = 0
		for j in range(i):
			if lst[j] < num and max_seq < res[j]:
				max_seq = res[j]
		res.append(max_seq + 1)
	# Subtract 1 to have numbers in range [0,N_CLASSES)
	for i in range(N_CLASSES):
		res[i] = res[i] - 1
	return res

# Get output for ce task
def get_ce(lst):
	res = []
	for i in range(N_CLASSES):
		ce_diff = MAXINT
		ce_val = MAXINT
		ce_ind = -1
		for j in range(N_CLASSES):
			if i == j:
				continue
			diff = abs(lst[i] - lst[j])
			if diff < ce_diff or (diff == ce_diff and lst[j] < ce_val):
				ce_diff = diff
				ce_val = lst[j]
				ce_ind = j
		res.append(ce_ind)
	return res

# Sample for Sorting a List
def gen_sort(dtype = 'int'):
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

# Sample for Longest Increasing Sequence
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
	# Subtract 1 to have numbers in range [0,N_CLASSES)
	for i in range(N_CLASSES):
		order[i] = order[i] - 1
	return lst, order

# Sample for ith element in the sorted list
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

# Order relations conversion
def order_relations(lst):
	c_lst = []
	for j in range(N_CLASSES - 1):
		for k in range(j+1, N_CLASSES):
			if lst[j] > lst[k]:
				c_lst.append(1)
			else:
				c_lst.append(0)
	return c_lst

# Relational table conversion
def rel_table(lst):
	mat = []
	for i in range(0,len(lst)):
		nlst = []
		for j in range(0,len(lst)):
			nlst.append(lst[i])
			nlst.append(lst[j])
		mat.append(nlst)
	return mat

# Relational table data
def rel_table_data(n_samples):
	lsts, mats, orders = list(), list(), list()
	for i in range(1,n_samples+1):
		lst, order = gen_sort()
		lsts.append(lst)
		mats.append(rel_table(lst))
		orders.append(order)
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, mats, orders

# Dataset for Sorting a List
def sort_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, order = gen_sort()
		lsts.append(lst)
		orders.append(order)
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, orders

# Dataset for Longest increasing sequence
def lis_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, order = gen_lis()
		lsts.append(lst)
		orders.append(order)
		if i % 1000 == 0:
			print("Generated " + str(i) + ' samples')
	return lsts, orders

# Dataset for Order Relations
def order_relations_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, order = gen_sort()
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

# Dataset for Numbers and Order Relations
def all_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, order = gen_sort()
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

# Dataset for Minimum
def simple_data(n_samples):
	lsts, orders = list(), list()
	for i in range(1,n_samples+1):
		lst, mins = gen_ith(1)
		lsts.append(lst)
		orders.append(mins)
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
		return sort_data(n_samples)
	elif data_type == "order_relations":
		print ("ORDER RELATIONS")
		return order_relations_data(n_samples)
	elif data_type == "all":
		print ("ALL DATA")
		return all_data(n_samples)
	elif data_type == "rel_table":
		print ("RELATIONS TABLE")
		return rel_table_data(n_samples)

def data_by_task_and_form(task, form, is_training = True):
	n_samples = N_SAMPLES
	n_samples = n_samples if is_training else int(n_samples / 5) # 20% of samples for validation
	lsts, inps, rslts = [], [], []
	for i in range(1,n_samples+1):
		lst, res = gen_list(), []
		if task == "sort":
			res = get_sort(lst)
		elif task == "lis":
			res = get_lis(lst)
		elif task == "ce":
			res = get_ce(lst)
		inp = []
		if form == "lst":
			inp = lst
		elif form == "order_rel":
			inp = order_relations(lst)
		elif form == "all":
			inp = lst + order_relations(lst)
		elif form == "rel_table":
			inp = rel_table(lst)
		lsts.append(lst)
		inps.append(inp)
		rslts.append(res)
	if form == "lst" or form == "order_rel" or form == "all":
		return inps, rslts
	return lsts, inps, rslts

def test():
	lst = gen_list()
	sort = get_sort(lst)
	lis = get_lis(lst)
	ce = get_ce(lst)
	pdb.set_trace()

# test()