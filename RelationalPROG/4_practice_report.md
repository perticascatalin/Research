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

# Defining the neural network
def neural_net(x, num_classes, num_labels, reuse, is_training):
	with tf.variable_scope('NeuralNet', reuse = reuse):
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

...
```

#### 10.2 Tasks

#### 10.3 Frameworks

##### Tensorflow

###### V1

**Functions:**

```python
tf.variable_scope
tf.layers.dense
tf.layers.dropout
tf.nn.softmax
```

###### V2

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

Finally, managed to run the code for extracting features (using python2 because python 3 has tf version too much ahead), but since it's not running on GPUs, this process can be quite lenghty, estimated at around 4 hours for the whole dataset. So I limit the number of images from 70000 to 3000 just to see what happens.

`python2 main.py --expName "clevrExperiment" --train --trainedNum 3000 --testedNum 3000 --epochs 25 --netLength 4 @configs/args.txt`

Need to improve some parts of the code because at the training step we're getting an error for trying to access samples which are not within the extracted 3000.