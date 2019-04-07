from scipy.special import entr

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

print len(lsts_val)
print len(orders_val)

#print lsts_val[0]
#print orders_val[0]

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

probability_bin(lsts_val)
probability_base(orders_val)