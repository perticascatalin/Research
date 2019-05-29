
def bit_comparator(x, inputs, n_classes, num_labels, dropout, reuse, is_training):
	with tf.variable_scope('BitComparator', reuse = reuse):

		# Define unit neuron
		fc = tf.layers.dense(x, 1, activation = tf.nn.sigmoid)
		
		# Define output
		output = tf.nn.softmax(fc) if not is_training else output

	return output, inputs