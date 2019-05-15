


def relational_net(x, inputs, n_classes, num_labels, dropout, reuse, is_training):
	with tf.variable_scope('RelationalNet', reuse = reuse):

		units_1 = []
		for i in range(n_classes):
			for j in range(n_classes):
				# Combine 2 input units into a relational unit
				# This is pseudocode for now, but could be 2 possible ways of writing
				rel_unit = tf.layers.dense([inputs[:,i], inputs[:,j]], 1)
				rel_unit = tf.layers.dense(tf.concat([inputs[:,i], inputs[:,j]], 1), 1)


		# # Define first layer
		# fc = tf.layers.dense(x, layer_neurons[0], activation = tf.nn.tanh)
		# if num_layers >= 2:
		# 	fc = tf.layers.dropout(fc, rate = layer_dropout[0], training = is_training)
		# layers = [fc]

		# # Define hidden layers
		# for i in range(1, num_layers - 1):
		# 	last_layer = layers[-1]
		# 	fc = tf.layers.dense(last_layer, layer_neurons[i], activation = tf.nn.tanh)
		# 	fc = tf.layers.dropout(fc, rate = layer_dropout[i], training = is_training)
		# 	layers.append(fc)

		# # Define last layer
		# if num_layers >= 2:
		# 	last_layer = layers[-1]
		# 	fc = tf.layers.dense(last_layer, layer_neurons[-1], activation = tf.nn.tanh)

		# # Define outputs
		# outputs = list()
		# for i in range(n_classes):
		# 	out_i = tf.layers.dense(fc, num_labels)
		# 	out_i = tf.nn.softmax(out_i) if not is_training else out_i
		# 	outputs.append(out_i)

	return outputs, inputs

inputs = [[1,3,2],[3,2,1]]
inputs[]????