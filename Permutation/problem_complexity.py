import math
import numpy as np
import matplotlib.pyplot as plt

import analysis as co
import generator as gen
import setup as stp

n = stp.num_classes
data_type = stp.data_type

print 'num classes:', n
print 'data type:', data_type

if data_type == "data":
	lsts_val, orders_val = gen.data()
if data_type == "simple_data":
	lsts_val, orders_val = gen.simple_data()
elif data_type == "order_relations":
	lsts_val, orders_val = gen.order_relations()
elif data_type == "all":
	lsts_val, orders_val = gen.all()

print 'Num inputs:', len(lsts_val)
print 'Num outputs:', len(orders_val)

#print lsts_val[0]
#print orders_val[0]

# Trying to compute entropy for input
# Example:
# N = 5
# Number of possible inputs (theoretical):
# 50 * 49 * 48 * 47 * 46 = 254.251.200
# Probability of any input = 1/254.251.200

# num_pos = 254251200
# iprob = 0.00000000393312
# log_iprob = math.log(iprob, 2)

# print iprob
# print log_iprob

n_size = n

for n_size in range(1,6):
	print 'SIZE:', n_size
	print '============='
	num_pos = 1
	max_num = 50
	for i in range(n_size):
		num_pos *= max_num
		max_num -= 1
	iprob = 1.0/num_pos
	log_iprob = math.log(iprob, 2)

	print iprob
	print log_iprob

	entropy = -(iprob * log_iprob)
	print entropy


def probability_bin(lsts):
	# use base 2
	bins = {}
	for lst in lsts:
		num = 0
		for bit in lst:
			num = num * 2 + bit
		if num in bins:
			bins[num] = bins[num] + 1
		else:
			bins[num] = 1

	print bins

def probability_base(lsts):
	# use base n
	base = n
	bins = {}
	for lst in lsts:
		num = 0
		for nit in lst:
			num = num * base + nit
		if num in bins:
			bins[num] = bins[num] + 1
		else:
			bins[num] = 1

	print bins

def entropy(lsts):
	return 0

#probability_bin(lsts_val)
#probability_base(orders_val)

def lin(x):
	return math.log(x, 2)
	#return x

def arrangement(maxint, n_size):
	return 0

# returns array of factorials
def factorial(n_size):
	start = 1
	arr = [lin(1)]
	for i in range(n_size):
		start *= (i+1)
		arr.append(lin(start))
	return arr

# returns array of pow2s
def pow2(n_size):
	start = 1
	arr = [lin(1)]
	for i in range(n_size):
		start *= 2
		arr.append(lin(start))
	return arr

# approximate 2^(n^2) - still incorrect
def pow2comb(n_size):
	start = 1
	arr = [lin(1)]
	for i in range(n_size):
		n = (i+1)
		p = (n * (n-1))/2
		tn = pow2(p)
		start = tn[-1]
		arr.append(start)
	return arr

n = 8

index = np.arange(n+1)
fact = factorial(n)
p2 = pow2(n)
p2c = pow2comb(n)

print index
print fact
print p2
print p2c

plt.plot(index, fact, 'r', index, p2, 'g', index, p2c, 'b')
plt.show()
