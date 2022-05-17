## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 10. Report on practice (technical details)

#### 10.1 Models

##### 10.1.1 Multi-label Multi-class Neural Network

The model accepts N inputs and M outputs for training and performs standard classification tasks. This architecture could be viewed as modelling a fixed seq2seq task or as an array-like input (sample with N features) mapped to an array-like output consisting of various classes of labels (M classes, eg. color, shape and size for M = 3), each with its own set of labels (red, green and blue; small and large, etc.).

What is particularly useful in this case is the fact that we do not have to train separate models for each class (category). The second advantage is that we do not have to represent each class as a combination of labels. Eg. {blue, large and square} could represent one class, while {blue, small and circle} would represent another one, and thus the number of labels would grow too rapidly with the increase in the number of classes (categories). Yet a third advantage is that the model can robustly represent setups with input-output pairs for program induction.

*Implementation using python 2.7 and TensorFlow 1.15*

```python
import tensorflow as tf

layer_neurons = [512,256]
layer_dropout = [0.0]

# Defining the neural network
def neural_net(x, num_classes, num_labels, layer_neurons, layer_dropout, reuse, is_training):
	with tf.variable_scope('NeuralNet', reuse = reuse):
		# Define layers: first is input-dense with dropout, last is dense-classes no dropout
		num_layers = len(layer_neurons)
		layers = []
		for i in range(0, num_layers):
			last_layer = x if i == 0 else layers[-1]
			this_layer = tf.layers.dense(last_layer, layer_neurons[i], activation = tf.nn.tanh)
			this_layer = this_layer if i == num_layers - 1 else tf.layers.dropout(this_layer, rate = layer_dropout[i], training = is_training)
			layers.append(this_layer)
		# Define outputs (note: num_labels can differ by i)
		outputs = []
		for i in range(num_classes):
			out_i = tf.layers.dense(this_layer, num_labels)
			out_i = out_i if is_training else tf.nn.softmax(out_i)
			outputs.append(out_i)
	return outputs
```

##### 10.1.2A Relational Neural Network

One way of creating a relational network is to pair up elements from an input sample, concatenate them into a single vector and then link the vector to one or more neurons on the following layer in the neural network. The example below links the pairing vector to one neuron, thus creating N x N neurons in the second layer. After this, it applies the same convolutional filters to all rows in an attempt to represent each sample as an aggregation of its relations to other samples from the same input. Layers of neurons can be further applied, but in the example below we directly apply the softmax layer.

*Implementation using python 2.7 and TensorFlow 1.15*

```python
import tensorflow as tf

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
		units_3 = tf.expand_dims(tf.stack(units_2, axis = 2), 3)
		# Aggregate rows with a convolution
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
```

##### 10.1.2B Convolutionally Relational Neural Network

A variation of the previous neural network is to apply the same learning function (learn the same weights) to the pairs of elements, thus learning the same relations between all the elements. We can implement this as a convolutional filter, which drastically reduces the training time of the previous neural network and at the same time also improves the results in our experimental setup.

*Implementation using python 2.7 and TensorFlow 1.15*

```python
import tensorflow as tf

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
		units_1b = tf.layers.conv2d(units_1a, 8, [2,1], [2,1], 'same', activation = 'relu')
		# Aggregate rows with a convolution: results in a [batch_sz, 1, Nx1, 4] Tensor
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
```

#### 10.2 Tasks

##### 10.2.1 Sorting an array of elements

**Input**: Array of N unique elements (integers) with values in the range [1,50].

**Output**: Array of N values specifying the position of each element in the sorted array.

**Description**: This task is relational because the outputs depend on how large an element is in comparison to the rest of the elements in the input array.

![Accuracy for N = 30](https://raw.githubusercontent.com/perticascatalin/Research/master/PermutationRN/results/all_30_acc.png)

![Loss for N = 30](https://raw.githubusercontent.com/perticascatalin/Research/master/PermutationRN/results/all_30_loss.png)

![Accuracy all models, various N](https://raw.githubusercontent.com/perticascatalin/Research/master/PermutationRN/results/acc_all.png)

**Legend**: TODO

#### 10.3 Frameworks

##### Tensorflow

###### V1

**Functions:**

```python
tf.variable_scope

tf.layers.dense
tf.layers.dropout
tf.layers.conv2d

tf.nn.softmax
tf.contrib.layers.flatten

tf.slice
tf.concat
tf.expand_dims

```

###### V2

##### Keras

#### 10.4 Pre-Processing

##### AST

##### DOM

#### 10.5 Open Source Repositories

##### Relational Network

Sample open source [code](https://github.com/clvrai/Relation-Network-Tensorflow) applying RN to the Sort-of-CLEVR dataset.

This repository is deprecated since Tensorflow2 (attempted to migrate to TF2, but got stuck at `contrib.layers.optimize_loss`, no equivalent in TF2). Revert to TF1. This implies reverting to Python 3.6, was using Python 3.8. Got stuck with pyenv, cannot revert to 3.6. Finally tried using Python 2.7, with Tensorflow 1.15, got a working version this way. Plots visualized via Tensorflow-Plot.

- `sudo python2/python3 -m pip install package` (for python 2.7/3.8)
- `python2 generator.py`
- `python2 trainer.py`
- `tensorboard --logdir ./train_dir`

Tensorboard then provides us with a localhost link and we can look at training statistics in the browser:

![Tensorboard](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/tensorboard.png)

After 15.000 training steps, we take a look at the currently tested images and we can see that 3 of 4 questions were answered correctly:

![Sample images](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/samples.png)

After another 15.000 training steps the accuracy on the testing data reaches an average of 95%, while the loss drops to around 0.1:

![Accuracy and Loss](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/loss_acc.png)

This test run was performed on a dataset containing images with 4 objects. The images used have 128 x 128 pixels of various colors.

The sizes of the images and the number of objects can be customized. The model's performance can be compared to baseline MLP/CNN models which do not use a relational function.

**Note**: the evaluated implementation of the RN model does not process question embeddings.

##### MAC Network

Original open source [code](https://github.com/stanfordnlp/mac-network) implementation of MAC network.

Encountered some problems with `imread` from `scipy.misc` when trying to extract the ResNet-101 features prerequisite for training. Probably solved, using `from PIL import Image` and replacing the reading and resizing functions accordingly.

Needed to install torch and torchvision. The code cannot run with tourch installed without CUDA enabled. Solved by removing `model.cuda()`.

Checking a package version: eg. `import tensorflow as tf`, `tf.__version__`.

Finally, managed to run the code for extracting features (using python2 because python 3 has tf version too much ahead), but since it's not running on GPUs, this process can be quite lenghty, estimated at around 4 hours for the whole dataset.

Dataset samples:

|Sample 1|Sample 2|
|:------:|:------:|
|![Img 1](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/clevr_1.png)|![Img 2](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/clevr_2.png)

Question words mapped to integers which can be one-hot encoded later on.

![Word vector](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/word_vector.png)

So I limit the number of images from 70000 to 3000 just to see what happens.

`python2 main.py --expName "clevrExperiment" --train --trainedNum 3000 --testedNum 3000 --epochs 25 --netLength 4 @configs/args.txt`

Need to improve some parts of the code because at the training step we're getting an error for trying to access samples which are not within the extracted 3000.