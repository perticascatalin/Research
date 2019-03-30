# For instance N = 10 (wrt position)
# Inputs   1 2 3 4 5 6 7 8 9 10
# Generate 2 4 3 1 8 9 6 7 5 10
num_classes = 10
num_out_classes = num_classes

# Number of arrays to generate
# Depending on num_classes, the num_samples should be above a certain threshold for decent accuracy
# The threshold can be computed using N! = num_classes!
# N!/num_samples < T
# 6! = 720
# 1000 decent sample
# 8! = 40320
# 10000 decent sample
# 10! = 3628800
# 60000 decent sample
num_samples = 60000

# Maximum number in array
maxint = 50

# Number of estimators for decision tree forests
n_estim = 96

# Layer neurons for neural network
layer_neurons = [512, 256, 128]

# The type of data
# Can be "data", "order_relations", "all", or "simple_data" for min/max (change ith)
data_type = "data"

if data_type == "simple_data":
	num_out_classes = 1