## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 5. Report on Bibliography

This report continues the study started in the [thesis proposal](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/exec_abstract.md) by developing an in-depth analysis of the selected bibliography. The review on related literature is carried out with regards to our previously set research goals.

**Overall Goals:**

- Research possible ways to integrate relational reasoning in deep learning models
- Investigate the role of relational reasoning in solving programming tasks

**Content**

[1. A simple Neural Network Model for Relational Reasoning](https://arxiv.org/pdf/1706.01427.pdf)

[2. Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf)

[3. Deep Coder: Learning to Write Programs](https://arxiv.org/pdf/1611.01989.pdf)

[4. Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/pdf/2006.08381.pdf)

[5. On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547.pdf)

### 6. Studies

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

|Image/Dataset|Question/Answer|
|:-----:|:-------------:|
|![Sort-of-CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/sort_of_clevr.png)| What is the color of the nearest object to the yellow object?|
|Sort-of-CLEVR|Green|
|![CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/clevr.png)|What size is the cylinder that is left of the brown metal thing that is left of the big sphere?|
|CLEVR|Small|

The second step is to examine how the make the object extraction question dependent. Questions are processed word-by-word (list of integers assigned to words) by an LSTM and the final state (an embedding of the question) is passed on to the RN together with the objects pair.

These triples are batched through an MLP for learning the relational function g (the output of which is a measure of relevance for object pairs). These outputs are then aggregated by a second MLP which learns the function f used for providing the answer to the question. Finally, the information about what constitues objects and how to parse the question is backpropagated in the CNN and in the LSTM. This approach shows a 27% improvement in accuracy on the CLEVR  dataset compared to the state-of-the-art at the time of the publication and a 31% improvement in accuracy on the Sort-of-CLEVR dataset compared to a baseline CNN-MLP architecture.

Sample open source [code](https://github.com/clvrai/Relation-Network-Tensorflow) applying RN to the Sort-of-CLEVR dataset.

##### 1.2 Code Running Logs

This repository is deprecated since Tensorflow2 (attempted to migrate to TF2, but got stuck at `contrib.layers.optimize_loss`, no equivalent in TF2). Revert to TF1. This implies reverting to Python 3.6, was using Python 3.8. Got stuck with pyenv, cannot revert to 3.6. Finally tried using Python 2.7, with Tensorflow 1.15, got a working version this way. Plots visualized via Tensorflow-Plot.

- `sudo python/python3 -m pip install package` (for python 2.7/3.8)
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

#### 3. Deep Coder: Learning to Write Programs

##### 3.1 High-Level Summary

This paper presents an approach to program induction involving the use of neural networks to predict the probability that certain methods (from a predefined DSL) appear in a program satisfying a set of given input-output constraints. These probabilities are then used to optimize the search for the program satisfying the input-output constraints.

Overall, the study showcases a potential solution to the IPS problem (Inductive Program Synthesis), called LIPS (Learning Inductive Program Synthesis). This approach can be split into 4 sub-parts:

- **1. DSL specifications and Program attributes**: the use of a DSL is generally necessary for the purpose of restricting the search space of programs (by abstracting away the technical details). Current program synthesis methods do not work well on general programming languages because of the combinatorial explosion of the search space. However, experiments have been performed on Python ASTs, such as (P. Yin & G. Neubig [2]), where a neural network is used to encode natural language specifications into a latent representation and then a second neural network decodes the representation into an AST. In the case of Deep Coder, the program attributes are represented by the estimated probabilities of a method from the DSL to appear in the target program. These attributes are then further used to reduce the search space.

- **2. Data generation**: the advantages of using DSLs are not limited to search space reduction. One can leverage DSLs to generate synthetic data which can be used for training. In the case of Deep Coder, the generated data consists of small programs which can be evaluated on random input data for the purpose of obtaining input-output pairs. The IO pairs and the program are then used in training a neural network to predict the program attributes used for narrowing down the search problem.

- **3. Machine Learning Model to predict Program Attributes**: M input-output examples (M = 5) are used as input for the network (these are padded with a special value in order to have a fixed length). Next, the encoder part of the network concatenates the input & output examples types, as well as their embeddings into a final program state representation vector. The different examples available are aggregated by averaging. Finally, the network outputs predictions for program attributes (probability that a function from the DSL will appear or will be absent in the program).

|Component|Illustration|
|:-------:|:----------:|
|Sample Program|![Program Example](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/prog.png)|
|MLP Architecture|![Model](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/dcnn.png)|
|MLP Output|![Program Attributes](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attrib.png)|

- **4. Search Strategy guided by estimated Program Attributes**: the study tests and compares 2 different search strategies (DFS, Sort & Add enumeration - a branch and bound type of strategy, Beam Search - a version of BFS), with and without guidance from the estimated program attributes. The approach based on program attributes shows considerable search performance improvements. The comparisons to other baseline methods - Beam Search & the SMT solver from (A. Solar-Lezama [3]) are made available. To note however that the program sizes are quite small, up to T = 5 instructions.

|Component|Illustration|
|:-------:|:----------:|
|Search Tree|![Search](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/g_search.png)|
|Search Times|![Times](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/dc_times.png)|

##### 3.2 Programs Representation

This section summarizes the scope of the DSL used in Deep Coder and briefly explains the complexity of using ASTs, which have the capability to represent general purpose programs and do not have to be synthetically generated, such as DSLs.

**Abstract Syntax Trees**

| Implementation   | # Lines | # Nodes | # Attributes |
|:----------------:|:-------:|:-------:|:------------:|
| Euclid Algorithm | 7       | 48      | 70           |
| Iteration        | 9       | 73      | 99           |

|Component|Illustration|
|:-------:|:----------:|
|Abstract Syntax Tree|![AST](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/ast.png)|

**Domain Specific Language**

The DSL defined in this study is comprised of first order and higher order functions.

*First Order Functions*

Head, Last, Take, Drop, ...

*Higher Order Functions*

Map, Filter, ...

Sample open source [code1](https://github.com/dkamm/deepcoder), [code2](https://github.com/HiroakiMikami/deep-coder) implementing Deep Coder.

### 7. Remarks

### 8. Definitions and Notes

1. **Relational Reasoning**: the capacity to reason about and find solutions to problems involving a set of objects which are related to one another through some properties that need to be discovered.

2. **Program Induction**: the inference of computer programs designed to solve problems given in the form of partial data, either input-output pairs, or other types of constraints.

3. **Feature Map**: a collection of kernel activations which are the result of applying filters from one convolutional network layer to the next layer. The filters / kernels are represented by learnable weights, while the feature map is the activation of a CNN at an intermediary layer.

4. **Embedding**: a relatively low-dimensional space into which high-dimensional vectors can be translated.

5. **Latent Representation**: a representation of data which is available in a neural network's hidden layers. These representations fill a latent space, which can be viewed as an embedding when the network acts as an encoder (when it compresses data). To note the type of embeddings which retain semantic properties, such as Word2Vec (Mikolov et al. [1]).

6. **Inductive Program Synthesis**: IPS problem, given input-output examples, produce a program that has behavior consistent with the examples. This requires solving 2 problems: defining the program space & the search procedure and solving the ranking problem - deciding which program is to be preferred when several solutions are available.

7. **Abstract Syntax Tree**: AST, a tree representation of the abstract syntactic structure of source code written in a programming language.

8. **Beam Search**: a version of BFS, which uses a heuristic to only keep a subset of (best) partial solutions explored at any given point during the search process.

### 9. Additional References

1. [Distributed Representations of Words and Phrases and their Compositionality, 2013](https://arxiv.org/pdf/1310.4546.pdf)

2. [A Syntactic Neural Model for General-Purpose Code Generation, 2017](https://arxiv.org/pdf/1704.01696.pdf)

3. [The Sketching Approach to Program Synthesis, 2008](https://people.csail.mit.edu/asolar/papers/Solar-Lezama09.pdf)