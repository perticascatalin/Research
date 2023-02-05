import tensorflow as tf

def conv_rel_net(rel_table, num_classes, num_labels, reuse, is_training):
    with tf.variable_scope('ConvRelNet', reuse = reuse):
        rel_units = tf.layers.conv2d(rel_table, 8, [1,2], [1,2], 'same', activation = 'relu')
        rel_units = tf.keras.activations.tanh(rel_units)
        agg_units = tf.layers.conv2d(rel_units, 4, [1,num_classes], [1,num_classes], 'same', activation = 'relu')
        feat_vect = tf.contrib.layers.flatten(agg_units)
        outputs = []
        for i in range(num_classes):
            out_i = tf.layers.dense(feat_vect, num_labels)
            out_i = out_i if is_training else tf.nn.softmax(out_i)
            outputs.append(out_i)
    return outputs