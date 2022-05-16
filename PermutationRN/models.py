import tensorflow as tf

def neural_net(x, num_classes, num_labels, layer_neurons, layer_dropout, reuse, is_training):
	with tf.variable_scope('NeuralNet', reuse = reuse):
		# Define layers: first is input-dense with dropout, last is dense-classes no dropout
		num_layers = len(layer_neurons)
		layers = []
		for i in range(0, num_layers):
			last_layer = x if i == 0 else layers[-1]
			this_layer = tf.layers.dense(last_layer, layer_neurons[i], activation = tf.nn.tanh) # Tanh: 8.8, Sigmoid: 6.6, Relu: X (no convergence)
			this_layer = this_layer if i == num_layers - 1 else tf.layers.dropout(this_layer, rate = layer_dropout[i], training = is_training)
			layers.append(this_layer)
		# Define outputs (note: num_labels can differ by i)
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(this_layer, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)
	return outputs