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

def norm_conv_rel_net(x, num_classes, num_labels, batch_size, reuse, is_training):
	with tf.variable_scope('NormConvRelNet', reuse = reuse):
		units_1 = []
		for i in range(num_classes):
			for j in range(num_classes):
				a_unit = tf.slice(x, [0,i], [batch_size,1])
				b_unit = tf.slice(x, [0,j], [batch_size,1])
				rel_unit = tf.concat([a_unit, b_unit], 1)
				units_1.append(rel_unit)
		units_2 = tf.stack(units_1, axis = 2)
		units_3 = tf.expand_dims(units_2, 3)
		units_4 = tf.layers.conv2d(units_3, 8, [2,1], [2,1], 'same', activation = 'relu')
		units_5 = tf.keras.activations.tanh(units_4)
		units_6 = tf.layers.conv2d(units_5, 4, [1,num_classes], [1,num_classes], 'same', activation = 'relu')
		units_7 = tf.contrib.layers.flatten(units_6)
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(units_7, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)
	return outputs

def conv_rel_net(x, num_classes, num_labels, batch_size, reuse, is_training):
	with tf.variable_scope('ConvRelNet', reuse = reuse):
		units_1 = tf.layers.conv2d(x, 8, [1,2], [1,2], 'same', activation = 'relu')
		units_2 = tf.keras.activations.tanh(units_1)
		units_3 = tf.layers.conv2d(units_2, 4, [num_classes,1], [num_classes,1], 'same', activation = 'relu')
		units_4 = tf.contrib.layers.flatten(units_3)
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(units_4, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)
	return outputs

def conv_relational_net(x, num_classes, num_labels, batch_size, reuse, is_training):
	with tf.variable_scope('ConvRelationalNet', reuse = reuse):
		units_1 = []
		for i in range(num_classes):
			for j in range(num_classes):
				# Combine 2 input units into a relational unit
				a_unit = tf.slice(x, [0,i], [batch_size,1])
				b_unit = tf.slice(x, [0,j], [batch_size,1])
				rel_unit = tf.concat([a_unit, b_unit], 1)
				units_1.append(rel_unit)
		# Stack and create last dim channel [batch_sz, 2, NxN, 1]
		units_1a = tf.expand_dims(tf.stack(units_1, axis = 2), 3)
		# Aggregate pairs with a convolution: results in a [batch_sz, 1, NxN, 8] Tensor
		# num_channels: 8, filters, strides, same or valid
		# same or valid: "SAME will output the same input length, while VALID will not add zero padding"
		units_1b = tf.layers.conv2d(units_1a, 8, [2,1], [2,1], 'same', activation = 'relu')
		# Aggregate rows with a convolution: results in a [batch_sz, 1, Nx1, 4] Tensor
		# num_channels: 4, filters, strides, same or valid
		units_2 = tf.layers.conv2d(units_1b, 4, [1,num_classes], [1,num_classes], 'same', activation = 'relu')
		# Flatten: results in a [batch_sz, 4*N] Tensor
		units_3 = tf.contrib.layers.flatten(units_2)
		# Define outputs: N softmaxes with N classes
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(units_3, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)
	return outputs
	# Additional trials
	# init = tf.random_uniform_initializer(minval = -1.0, maxval = 1.0)
	# units_1b = tf.layers.conv2d(units_1a, 8, [2,1], [2,1], 'same', activation = 'relu', kernel_initializer = init)
	# units_2 = tf.layers.conv2d(units_1b, 4, [1,num_classes], [1,num_classes], 'same', activation = None)
	# units_2 = tf.layers.average_pooling2d(units_1b, [1,num_classes], [1,num_classes], 'same')

def relational_net(x, num_classes, num_labels, batch_size, reuse, is_training):
	with tf.variable_scope('RelationalNet', reuse = reuse):
		units_1 = []
		for i in range(num_classes):
			for j in range(num_classes):
				# Combine 2 input units into a relational unit
				a_unit = tf.slice(x, [0,i], [batch_size,1])
				b_unit = tf.slice(x, [0,j], [batch_size,1])
				rel_unit = tf.layers.dense(tf.concat([a_unit, b_unit], 1), 1, activation = tf.nn.sigmoid)
				units_1.append(rel_unit)
		# Combine a sequence of units into an aggregator unit
		units_2 = []
		for i in range(num_classes):
			agg_unit = tf.concat(units_1[i*num_classes:(i+1)*num_classes], 1)
			units_2.append(agg_unit)
		# Stack and create last dim channel 
		# For conv layer the format is [batch_sz, height, width, channels]
		# ([batch_sz, N, N, 1])
		units_3 = tf.expand_dims(tf.stack(units_2, axis = 2), 3)
		# Aggregate with a convolution
		# num_channels: 4
		# filter, strides
		# same or valid "SAME will output the same input length, while VALID will not add zero padding"
		units_4 = tf.layers.conv2d(units_3, 4, [1,num_classes], [1,num_classes], 'same', activation = 'relu')
		# Flatten: will result in [batch_sz,4*N] Tensor
		units_5 = tf.contrib.layers.flatten(units_4)
		# Define outputs: N softmaxes with N classes
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(units_5, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)
	return outputs

def net_inspect(x, num_classes, num_labels, batch_size, reuse, is_training):
	with tf.variable_scope('NetInspect', reuse = reuse):
		units_1 = []
		for i in range(num_classes):
			for j in range(num_classes):
				a_unit = tf.slice(x, [0,i], [batch_size,1])
				b_unit = tf.slice(x, [0,j], [batch_size,1])
				rel_unit = tf.concat([a_unit, b_unit], 1)
				units_1.append(rel_unit)
		units_1a = tf.expand_dims(tf.stack(units_1, axis = 2), 3)
		units_1a = tf.transpose(units_1a, (0, 2, 1, 3))
		units_1b = tf.layers.conv2d(units_1a, 32, [1,2], [1,2], 'same', activation = 'relu')
		units_2 = tf.layers.conv2d(units_1b, 16, [num_classes,1], [num_classes,1], 'same', activation = 'relu')
		units_3 = tf.contrib.layers.flatten(units_2)
		units_4 = tf.layers.dense(units_3, 64, activation = tf.nn.tanh)
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(units_4, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)
	return units_1, units_1a, units_1b, units_2, units_3
	# return outputs