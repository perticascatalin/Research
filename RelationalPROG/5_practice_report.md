# Relational reasoning in deep learning: a parallel between solving visual and programming tasks

## Experimental work and technical details

We present a series on experiments concerning the integration of relational reasoning into neural networks. Their objective is to explore ways of embedding relational prior knowledge into the design of machine learning models. We start by applying relational priors to simple specific (relational) tasks which out-of-the-box learning models have difficulties learning and demonstrate experimentally the efficency of exploiting these priors.

...

**Content**

**10. Neural Problem Solving**

**11. Graph Neural Networks**

**12. Extraction of Visual Relationships**

**13. Data Representations in Programming**

**14. Machine Learning Frameworks**

## 10. Neural Problem Solving



### 10.1 Tasks

#### 10.1.1 Sorting a List

**Description**: Sorting a list of numbers is one of the first examples in every introductory computer science class. The task requires rearranging the elements of a list in increasing (or decreasing) order. There are numerous algorithms designed to solve this problem for any given input list. They differ in their strategies (eg. iterative or recursive) and their time complexity. For our purposes, this task is relational because the outputs depend on how large an element is in comparison to the rest of the elements in the input array.

**Input**: Array of N unique elements (integers) with values in the range [1,50].

**Output**: Array of N values denoting the position of each element in the sorted array.

**EXAMPLE**:

- 8 out of 10
- input: 25 19 26 30 16 40 21 23 39 41
- expect: 5 2 6 7 1 9 3 4 8 10
- pred:   6 2 5 7 1 9 3 4 8 10

**Explanation**:

#### 10.1.2 Longest Increasing Sequence

**Description**: Given a list of numbers, we are required to find the longest subset of numbers which are in increasing order given their initial positions. This task is one of the most simple dynamic programming examples, where the solution to a subproblem - longest sequence up to the ith element (best[i]) is computed based on the previously solved subproblems - longest sequences for 1st, 2nd,... (i-1)th elements (best[1..(i-1)]) provided their values are smaller. For this reason, this problem could also be viewed as a task where relational reasoning is required.

**Input**: Array of N unique elements (integers) with values in the range [1,50].

**Output**: Array of N values specifying the size of the longest increasing sequence ending with the current element.

**EXAMPLE**:

- 9 out of 10
- input: 25 19 26 30 16 40 21 23 39 41
- expect: 1 1 2 3 1 4 2 3 4 5
- pred:   1 1 2 2 1 4 2 3 4 5

**Explanation**:

### 10.2 Models

#### 10.2.A Multi-label Multi-class Neural Network

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

#### 10.2.B Relational Neural Network

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

#### 10.2.C Convolutional Relational Neural Network

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

### 10.3 Evaluation of Results

**DATASET INFO**:

- **Training**: 60.000 samples
- **Validation**: 12.000 samples
- **Training Iterations**: 100.000 steps (for neural networks only)

The accuracy is computed by averaging the number of correctly guessed labels per sample from the validation dataset. Eg. for 3 samples: 6 out of 10, 7 out of 10, 8 out of 10, then the model accuracy would report an accuracy of 70%.

#### 10.3.A Data Generation, Loss Function and Training

```python
lsts_train, orders_train = gen.data_by_type(data_type, is_training = True)
lsts_train = tf.convert_to_tensor(lsts_train, dtype = tf.float32)
orders_train = tf.convert_to_tensor(orders_train, dtype = tf.int32)

lsts_train, orders_train = tf.train.slice_input_producer([lsts_train, orders_train], shuffle = True)
X, Y = tf.train.batch([lsts_train, orders_train], batch_size = batch_size, capacity = batch_size * 8, num_threads = 4)
logits_train = mod.neural_net(X, N_INPUTS, N_OUTPUTS, layer_neurons, layer_dropout, reuse = False, is_training = True)
logits_test  = mod.neural_net(X, N_INPUTS, N_OUTPUTS, layer_neurons, layer_dropout, reuse = True,  is_training = False)

train_loss_op = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUTPUTS):
	train_loss_op = train_loss_op + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	logits = logits_train[i], labels = Y[:,i]))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(train_loss_op)

correct_pred_train = tf.constant(0.0, dtype = tf.float32)
for i in range(N_OUTPUTS):
	correct_pred_train = correct_pred_train + tf.cast(tf.equal(tf.argmax(logits_test[i], 1), tf.cast(Y[:,i], tf.int64)), tf.float32)
accuracy_train = tf.reduce_mean(correct_pred_train)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(1, num_steps + 1):
		sess.run(train_op)
```

#### 10.3.1 Sorting a List

**EXAMPLE**:

- 8 out of 10
- input: 25 19 26 30 16 40 21 23 39 41
- expect: 5 2 6 7 1 9 3 4 8 10
- pred:   6 2 5 7 1 9 3 4 8 10

|Sample|Accuracy by N|
|:----:|:-----------:|
|![Sort Lables](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/sort_labels.png)|![Accuracy all models, various N](https://raw.githubusercontent.com/perticascatalin/Research/master/PermutationRN/results/acc_all.png)|

|Accuracy|Loss|
|:------:|:--:|
|![Accuracy for N = 30](https://raw.githubusercontent.com/perticascatalin/Research/master/PermutationRN/results/all_30_acc.png)|![Loss for N = 30](https://raw.githubusercontent.com/perticascatalin/Research/master/PermutationRN/results/all_30_loss.png)|

**LEGEND**:

|Model|Code|Description|N=10|N=15|N=20|N=25|N=30|
|:---:|:--:|:---------:|:--:|:--:|:--:|:--:|:--:|
|Baseline    |base_data|Neural Net with 3 layers (512, 256, 128), using array as input (10.2.A)|100%|100%| 69%| 56%| 29%|
|Order Rel   |base_or  |Same Neural Net as the Baseline, using order relations as input        |100%|100%| 99%| 87%| 38%|
|CR Net      |C        |                                                                       |100%|100%| 98%| 98%| 86%|
|Norm CR Net |R_r      |Relational Net with paired inputs, convolute relations & norm output   |100%|100%|100%| 84%| 79%|
|Conv Rel Net|R        |Relational Net with paired inputs, convolute relations (10.2.C)        |100%| 94%| 81%| 75%| 80%|
|Rel Net     |Q        |Relational Net with paired inputs, fully connected (10.2.B)            | 97%| 58%| 49%| 45%| 44%|
|DT Order Rel|-        |Decision Trees using order relations as input                          | 81%| 42%| 25%| 16%| 12%|
|DT Baseline |-        |Decision Trees using array as input                                    | 55%| 34%| 25%| 20%| 16%|

#### 10.3.2 Longest Increasing Sequence

**EXAMPLE**:

- 9 out of 10
- input: 25 19 26 30 16 40 21 23 39 41
- expect: 1 1 2 3 1 4 2 3 4 5
- pred:   1 1 2 2 1 4 2 3 4 5

**LEGEND**:

|Model|Description|N=10|N=15|N=20|N=25|N=30|
|:---:|:---------:|:--:|:--:|:--:|:--:|:--:|
|Baseline    |(10.2.A) |98%|85%|74%|64%|57%|
|Conv Rel Net|(10.2.C) |89%|71%|64%|65%|64%|

## 11. Graph Neural Networks

### 11.1 Graph Convolutional Neural Network

[Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations/)

**Dataset**: Cora with paper subjects, words and citation links

**Problem**: Determine the subject (7 categories) of each paper based on the its word vector and citations (represented as a graph)

|Dataset samples|Dataset visualization|
|:-------------:|:-------------------:|
|![Cora samples](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/gnn_cora_cols.png)|![Cora visualization](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/gnn_cora_vis.png)|

**Baseline**: 73,5%

![Baseline](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/cora_baseline.png)

**GNN**: 83,3%

![GNN](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/cora_gnn.png)

*Implementation using python 3.9 and TensorFlow 2.8*

### 11.2 Relational Graph Convolutional Neural Network

TODO

## 12. Extraction of Visual Relationships

**Open Source Repositories**

### 12.1 Relational Network (Santoro et al. [5])

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

### 12.2 MAC Network (Hudson & Manning [4])

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

## 13. Data Representations in Programming

### 13.1 Abstract Syntax Tree (AST)

TODO

### 13.2 Document Object Model (DOM)

TODO

## 14. Machine Learning Frameworks

[Tools for drawing NN architectures](https://www.kaggle.com/getting-started/253300)

### Tensorflow / Keras

#### Tensorflow V1

**Functions:**

```python
tf.global_variables_initializer
tf.variable_scope
tf.constant
tf.convert_to_tensor
tf.reduce_mean

tf.train.slice_input_producer
tf.train.batch
tf.train.AdamOptimizer

tf.layers.dense
tf.layers.dropout
tf.layers.conv2d
tf.contrib.layers.flatten

tf.nn.softmax

tf.slice
tf.concat
tf.expand_dims

```

#### Tensorflow V2 / Keras

TODO