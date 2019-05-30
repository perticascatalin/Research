# For instance N = 10 (wrt position)
# Inputs   1 2 3 4 5 6 7 8 9 10
# Generate 2 4 3 1 8 9 6 7 5 10

# INPUT NUMBER OF CLASSES
#num_classes = 32 # (fully relational experiment)
#num_classes = 30 # (by design experiment)
#num_classes = 2 # (comparator experiment)
num_classes = 10 # (baseline experiment)

# OUTPUT NUMBER OF CLASSES
num_out_classes = num_classes

# Number of arrays to generate

num_samples = 60000
#num_samples = 10000
#num_samples = 2000 (comparator experiment)

# Maximum number in array
maxint = 50

# Number of estimators for decision tree forests
n_estim = 96

# Layer neurons and their dropout for neural network

# Uncomment to set baseline NN
#layer_neurons = [512, 256, 128]
#layer_dropout = [0.8, 0.8]

# Uncomment to set by design NN
layer_neurons = [400, 200] #(typical for 20 scale)
#layer_neurons = [1000, 200]
layer_dropout = [0.0]

# Uncomment to set single layer NN
#layer_neurons = [400] (typical for 20 scale)
#layer_neurons = [20]
#layer_dropout = []

# The type of data
# 1. "data"
# 2. "order_relations"
# 3. "all"
# 4. "simple_data" for min/max (change ith)
# 5. "comparator"
data_type = "data"

# Exception for number of output classes
if data_type == "simple_data" or data_type == "comparator":
	num_out_classes = 1