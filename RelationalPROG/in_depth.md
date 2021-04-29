## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### Report on Bibliography

This report continues the study started in the [thesis proposal](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/exec_abstract.md) by developing an in-depth analysis of the selected bibliography. The review on related literature is carried out with regards to our previously set research goals.

**Overall Goals:**

- Research possible ways to integrate relational reasoning in deep learning models
- Investigate the role of relational reasoning in solving programming tasks

**Content**

[1. A simple Neural Network Model for Relational Reasoning](https://arxiv.org/pdf/1706.01427.pdf)

[2. Deep Coder: Learning to Write Programs](https://arxiv.org/pdf/1611.01989.pdf)

[3. Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf)

[4. Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/pdf/2006.08381.pdf)

[5. On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547.pdf)

### Studies

#### 1. A simple Neural Network Model for Relational Reasoning

##### 1.1 High-Level Summary

This study presents a general machine learning model used for solving relational tasks. The relational network (RN) model contains a built-in mechanism for capturing core common properties of relational reasoning. From this perspective, we can draw parallels to other machine learning models designed with different properties in mind, such as the well known:

- CNNs used for extracting spatial and translation invariant properties
- LSTMs used for extracting sequential dependencies

The RN is used on 3 different tasks, all of which require some kind of relational inference on a set of objects:

- Visual question answering (datasets: CLEVR - 3d, Sort-of-CLEVR - 2d)
- Text-based question answering
- Complex reasoning about dynamic physical systems

Other similar approaches (relation-centric) include:

- Graph neural networks
- Gated graph sequential neural networks
- Interaction networks

|Equation|Note|
|:------:|:--:|
|![RN Formula](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/formula.png)|An important aspect is that the RN is a plug & play module, meaning that it can be placed on top of other deep learning models.|

- O set of objects {o1, o2, ..., on}
- f overall network output (across all pairs of objects)
- g outputs a relation between 2 objects (same for all pairs)
- f, g can be MLPs
- RN end-to-end differentiable

For our purposes, we will focus on how the visual question answering task is performed. Thus, the first step is to examine what constitutes an object in the context of RNs.

Since we are dealing with images (2-dimensional matrices), the objects could be sets of pixels extracted from the image. However, the RN model is intended to work on general objects. This means that we should not enforce extracting a region from an image and labeling it as an object because it would impose the definition of the object to be limited to the results provided by the region extraction algorithm.

The way this is dealt with is by using image embeddings. The images are processed using CNNs which create feature maps internally. The objects are then represented by feature map vectors. Assuming k feature maps (filters), each with d x d pixels, then one object is comprised of k values corresponding to the same location pixel in every feature map.

Thus, an object could comprise:

- the background
- particular physical objects
- a texture
- conjuctions of physical objects

The second step is to examine how the make the object extraction question dependent. Questions are processed word-by-word (list of integers assigned to words) by an LSTM and the final state (an embedding of the question) is passed on to the RN together with the objects pair.

From my understanding, these triples are batched through an MLP for learning a relational function (which pairs of objects are relevant), then these are aggregated in a second MLP that provides the answer to the question. Finally, the information about what constitues objects and how to parse the question is backpropagated in the CNN and in the LSTM.

Sample open source [code](https://github.com/clvrai/Relation-Network-Tensorflow) applying RN to the Sort-of-CLEVR dataset.

|Image/Dataset|Question/Answer|
|:-----:|:-------------:|
|![Sort-of-CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/sort_of_clevr.png)| What is the color of the nearest object to the yellow object?|
|Sort-of-CLEVR|Green|
|![CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/clevr.png)|What size is the cylinder that is left of the brown metal thing that is left of the big sphere?|
|CLEVR|Small|

##### 1.2 Code Running Logs

This repository is deprecated since Tensorflow2 (attempted to migrate to TF2, but got stuck at `contrib.layers.optimize_loss`, no equivalent in TF2).

Revert to TF1. This implies reverting to Python 3.6, was using Python 3.8. Got stuck with pyenv, cannot revert to 3.6.

Finally tried using:

- `sudo python -m pip install package` (for python 2.7)
- `sudo python3 -m pip install package` (for python 3.8)

Should use Python 2.7, with Tensorflow 1.15, got a working version this way. Plots visualized via Tensorflow-Plot.

- `python generator.py`
- `python trainer.py`
- `tensorboard --logdir ./train_dir`

Tensorboard then provides us with a localhost link and we can look at training statistics in the browser:

![Tensorboard](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/tensorboard.png)

After 15.000 training steps, we take a look at the currently tested images and we can see that 3 of 4 questions were answered correctly:

![Sample images](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/samples.png)

After another 15.000 training steps the accuracy on the testing data reaches an average of 95%, while the loss drops to around 0.1:

![Accuracy and Loss](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/loss_acc.png)

This test run was performed on a dataset containing images with 4 objects. The images used have 128 x 128 pixels of various colors. The sizes of the images and the number of objects can be customized. The model's performance can be compared to baseline MLP/CNN models which do not use a relational function.

#### 2. Deep Coder: Learning to Write Programs

##### 2.1 High-Level Summary

This paper presents an approach to program induction involving the use of neural networks to predict the probability that certain methods (from a predefined DSL) appear in a program satisfying a set of given input-output constraints. These probabilities are then used to optimize the search for the program satisfying the input-output constraints.

**First Order Functions**

Head, Last, Take, Drop, ...

**Higher Order Functions**

Map, Filter, ...

**Neural Networks**

- M input-output examples used as input for the network
- network outputs predictions for program attributes (probability that a function from the DSL will appear in the program)

**Search**

### Concluding Remarks

### Definitions

1. **Relational Reasoning**: the capacity to reason about and find solutions to problems involving a set of objects which are related to one another through some properties that need to be discovered.

2. **Program Induction**: the process of generating an executable program for solving a problem which is given in the form of input-output pairs, or other types of constraints.

3. **Feature Map**: a collection of kernel activations which result in a convolutional network layer by applying filters to the previous layer. The filters / kernels are represented by learnable weights, while the feature map is the output of a CNN at an intermediary layer.

4. **Embedding**: a relatively low-dimensional space into which high-dimensional vectors can be translated.

5. **Latent Representation**: a representation of data which is available in a neural network's hidden layers. These representations fill a latent space, which can be viewed as an embedding when the network acts as an encoder (when it compresses data). To note the type of embeddings which retain semantic properties, such as Word2Vec (Mikolov et al. [1]).

### Additional Bibliography

1. [Efficient Estimation of Word Representations in Vector Space, 2013](https://arxiv.org/pdf/1301.3781.pdf)