# For instance N = 10 (wrt position)
# Inputs   1 2 3 4 5 6 7 8 9 10
# Generate 2 4 3 1 8 9 6 7 5 10
num_classes = 8
num_out_classes = num_classes

# Number of arrays to generate
num_samples = 1000

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