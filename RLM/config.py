# INPUT NUMBER OF CLASSES
# num_inputs = 30 # (relational experiment)
# num_inputs = 20 # (baseline experiment)
# num_inputs = 3  # (counter experiment)
# num_inputs = 2  # (comparator experiment)
num_inputs = 20

# OUTPUT NUMBER OF CLASSES
num_outputs = num_inputs

# Number of arrays to generate
# num_samples = 60000 # (baseline experiment)
# num_samples = 8000  # (counter experiment)
# num_samples = 2000  # (comparator experiment)
#num_samples = 60000
num_samples = 10000

maxint = 50  # Maximum number in array
n_estim = 96 # Number of estimators for decision tree forests

# Layer neurons and their dropout for neural network

# Baseline NN
layer_neurons = [512, 256, 128]
layer_dropout = [0.0, 0.0]

# The type of data
# 1. "simple_data" for min/max (change ith)
# 2. "data"
# 3. "order_relations"
# 4. "all"
# 5. "comparator"
# 6. "counter"
data_type = "data"

# Data by task and format
# task from {"sort", "lis", "ce"}
# form from {"lst", "order_rel", "all", "rel_table"}
task = "sort"
form = "rel_table"

# Exception for number of output classes
if data_type == "simple_data" or data_type == "comparator" or data_type == "counter":
	num_outputs = 1