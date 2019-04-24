# For instance N = 10 (wrt position)
# Inputs   1 2 3 4 5 6 7 8 9 10
# Generate 2 4 3 1 8 9 6 7 5 10
num_classes = 20
num_out_classes = num_classes

# Number of arrays to generate
# Depending on num_classes, the num_samples should be above a certain threshold for decent accuracy
# The threshold can be computed using N! = num_classes!
# N!/num_samples < T
# 6! = 720
# 1000 decent sample (on par)
# 8! = 40320
# 10000 decent sample (generalize full for 4 classes)
# 10! = 3628800
# 60000 decent sample (generalize well for 60 classes)
# 20!/100.000 = 2.4e+13

# 12! = 479001600
# what is a decent sample?
# 61% accuracy with default params (last decent accuracy)
# with current dataset (generalize mah for 8000 classes)
num_samples = 60000

# Maximum number in array
maxint = 50

# Number of estimators for decision tree forests
n_estim = 96

# Layer neurons for neural network
# layer_neurons = [512, 256, 128]
# layer_dropout = [0.8, 0.8]
# layer_neurons = [400, 200]
layer_neurons = [800, 400]
layer_dropout = [0.0]

# The type of data
# Can be "data", "order_relations", "all", or "simple_data" for min/max (change ith)
data_type = "order_relations"

if data_type == "simple_data":
	num_out_classes = 1