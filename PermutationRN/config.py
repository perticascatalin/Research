# INPUT NUMBER OF CLASSES
#num_inputs = 30 # (relational experiment)
#num_inputs = 20 # (baseline experiment)
#num_inputs = 3  # (counter experiment)
#num_inputs = 2  # (comparator experiment)
num_inputs = 30

# OUTPUT NUMBER OF CLASSES
num_outputs = num_inputs

# Number of arrays to generate
num_samples = 60000 # (baseline experiment)
#num_samples = 8000  # (counter experiment)
#num_samples = 2000  # (comparator experiment)

maxint = 50  # Maximum number in array
n_estim = 96 # Number of estimators for decision tree forests

# Layer neurons and their dropout for neural network

# Baseline NN
#layer_neurons = [512, 256, 128]
#layer_dropout = [0.6, 0.6]

# Design NN
#layer_neurons = [400, 200] #(typical for N = 20)
#layer_neurons = [256, 160]
#layer_neurons = [100, 100]
layer_neurons = [900, 300]
layer_dropout = [0.0]

# Uncomment to set single layer NN
#layer_neurons = [400] #(typical for N = 20)
#layer_neurons = [20]
#layer_dropout = []

# The type of data
# 1. "data"
# 2. "order_relations"
# 3. "all"
# 4. "simple_data" for min/max (change ith)
# 5. "comparator"
# 6. "counter"
# 7. "lis"
data_type = "data"

# Exception for number of output classes
if data_type == "simple_data" or data_type == "comparator" or data_type == "counter":
	num_outputs = 1

# Model names and their description
# a_10 - baseline model data 10
# ac_10 - baseline model order_relations 10
# D_20,24,28,30 -> [400,200] with comparisons and N = 20,24,28,30
# E_30 -> [1000,200] with comparisons and N = 30
# F_30,20 flat 1 layer -> [1000][400] and N = 30, 20
# G_30,20 flat 1 layer -> [30][20] and N = 30, 20