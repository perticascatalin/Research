# MastersExperiments

Thesis Title: **Compositionality of relational operations in neural networks**

Objective: This document contains a brief introduction, the results from the first experiments (1/2) and the bibliography of the Masters Thesis.

## E1: Order relations compositionality in a sequence of numbers 

(almost done 4.5/5)

**Table of Contents**:

**Section 1** offers background and formulates the problem. **Section 2** offers an overview of the models tested in the experiments. **Sections 3-7** discusses results with baseline models (neural nets and decision trees), data re-representation (different datasets), the problem of scalability and how this relates to the problem complexity. **Sections 8-10** propose different solutions for improving the scalability and learning efficiency and provides supporting results. **Section 11** extracts the conclusions from the experiments. **Section 12** situates the importance of the experimental results in the larger goal. The bibilography offers a concise view of the readings involved in conducting this project.

- **1. Introduction**
- **2. Models**
- **3. Results**
- **4. Metrics**
- **5. Results by Dataset**
- **6. Scalability**
- **7. Problem Complexity**
- **8. Design Model**
- **9. Relational Model**
- **10. SortNet Model**
- **11. Conclusions**
- **12. Future Work**
- Bibilography

### 1. Introduction

#### 1.1 Relevance and priority

Whatever tasks they have to model, neural networks often encounter the problem of modeling relevance and priority - how to assign 1st, 2nd, 3rd and so on to data chunks such that it answers some questions. Such an example is the spatial ordering of elements in a visual scene as shown below.

|Example|Description|
|:-----:|:---------:|
|![logico_visual](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/images/logico_visual.png)|Start from logico-visual tasks learned by a neural network and continue with set theory and an analysis to solve particular questions.|

Understanding how neural networks and other supervised learning models learn to answer such questions by processing large amounts of labeled data is of importance to any visual automated tasks from tracking objects of interest in scenes to self-driving cars. Previous work on how to model this learning task includes [C] and [D]. The two papers are conceptually different, but both of them use CLEVR dataset as benchmark. [C] presents a neural network module which is claimed to better model relational reasoning. [D] goes into the area of attention, memory and composition. However, what they have in common is the apparently hard task of learning composed functions. 

#### 1.2 Visual Tasks

For instace, when asking a question about the nearest object to the black sphere, a neural network model has to learn the function of identifying the blue sphere, take the result, look for the furthest object to it, then retrieve it and tell something about its properties, which is yet another task to learn. One may conclude that answering such a simple question is a task in 3 steps, or is a task composed of 3 sub-tasks.

Neural networks owe some of their success to the ability of learning higher level compositions. From this perspective, we look at the challenges that come with an increased number of objects (above 10 typically). Some of our experiments might be able to empirically prove that NNs do not learn the correct compositions all the time, especially when the number of conceptually relevant objects gets higher than ~10. However, this is something that can be overcome at least partially through relational representation. The task of sorting an array of numbers (size N) is very expressive for this purpose - although the end result has to satisfy the constraint that all numbers are in their correct place, the composition is not N-fold, but 2-fold instead (at any time we only need to know how to compare 2 elements).

|Example|Description|
|:-----:|:---------:|
|![relational_sequence_sets](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/images/relational_sequence_sets.png)|Data relations: a sequence representing a certain order of the objects - set w. order relation. The set of all permutations. Symmetric group (composition of f and g to visit whole group).|

#### 1.3 Mathematical Structures

In this study we run experiments on data with mathematical properties such as sets, permutations and groups. And then we look at the impact of their solving difficulty in practical applications. Along the way we mention previously encountered challenges in machine learning from which the problem at hand suffers, but the focus will be on how we can exploit its structure.

The 0-1 hypothesis in permutation representation states that any permutation can be generate through an f-g path of transformations from the identical permutation.

The special arrangement hypothesis assumes an arrangement of multiple neural networks specialized in recreating a specific subset of transformations such that an input can be forwarded through a path or a cycle such that a specific state is always achieved.

#### 1.4 Predicting the correct order

The experiment models the manipulation of arrays with different numbers and of different lenghts. First, we formulate the problem of predicting the sorted order of the initial numbers. This problem involves the concepts of order relations, counting and permutations.

- N = 5
- MAX = 50
- distinct integers in range [1, MAX]

|Initial position|  1|  2|  3|  4|  5|
|:--------------:|:-:|:-:|:-:|:-:|:-:|
|Value           | 49|  3|  2|  5| 17|
|Sorted position |  5|  2|  1|  3|  4|
|Maximum         |  1|  0|  0|  0|  0|
|Minimum         |  0|  0|  1|  0|  0|

#### 1.5 Order Relations

*Total order relations are binary operators for which the value matrix has certain mathematical properties (a type of prior knowledge which can be exploited in statistical learning).*

- Order relations: Is A smaller than B?
- 0/1 for pair (A/B)

|  N| 49|  3|  2|  5| 17|
|:-:|:-:|:-:|:-:|:-:|:-:|
| 49|  -|  0|  0|  0|  0|
|  3|  1|  -|  0|  1|  1|
|  2|  1|  1|  -|  1|  1|
|  5|  1|  0|  0|  -|  1|
| 17|  1|  0|  0|  0|  -|

- Elementwise comparison vector:
- list entries from order matrix (left-right, top-bottom)
- [0 0 0 0 1 0 1 1 1 1 1 1 1 0 0 1 1 0 0 0]

In the above case, the sorted position of 49 is equal to O(1,1) + O(2,1) + O(3,1).... Similarly, the final position of 3 is equal to O(2,1) + O(2,2) + O(2,3).... Although the operations involved in finding the sorted positions are quite simple, statistical learning models have troubles with computing the correct answer.

#### 1.6 Concepts

**Concepts list:**

- information bottleneck (intro+ formula+ impl- debate-)
- scalability & problem complexity (intro+ formula+ impl+ debate+)
- learning complexity & separability (experiments+)
- compositionality - multiple learners (experiments+)
- relation to mathematical sets
- sequences

**Applications:**

- computer vision
- all problems involving permutations (even NP problems)
- rawest example: sequence of inputs, reorder inputs such that some function of the inputs in that particular order is maximized
- the number of possible permutations grows very fast with N and the possible number of order relations even faster.

#### Information Bottleneck

The information bottleneck principle links the analysis of deep neural networks to the domain of information theory. The key ideas of the principle are introduced as mathematical tools in a series of papers ([Z], [W]). These are: measuring the information propagation level against the theoretical upper bound and exploring the variation in information propagation during learning (the accuracy gain phase and the compression phase). How information is compressed inside a neural network depends on the data representation which can carry information about target outputs at different relevance levels. In practice, we can estimate the mutual information between random variables representing the input, the activation at a certain level and the output of the network. We can use these measures to intuitively analyze what is going through the network and then find out the factors which contribute to the success of the model design and parametrization.

#### Compositionality and Separability

Problems involving reasoning about the natural environment (eg. image analysis), as well as those involving structured data (eg. text and code) can often be expressed as series of operations applied to previous computations. In other terms, their solution can be expressed as a high level composition of functions.
These compositions can differ in their structure. For instance, CNNs use  kernel-based aggregations (hierarchical), while RNNs do recursive aggregations (sequential). Of course, these 2 approaches can be combined. What matters is how they help exploit the structure of the data, such that the computational load is reduced compared to using fully-connected neural networks.

### 2. Models

Briefly describe the models used and the intuitions behind them. Illustrate their structure and mechanism.

|Model name|Model description|
|:--------:|:---------------:|
|C-Design  |2-layer densely connected neural network with quadratic scaling|
|C-Baseline|3-layer densely connected neural network|
|C-1-Layer |1-layer densely connected neural network|
|C-Perceptron | perceptron for comparison operation |
|C-SortNet |model for algebraic sorting|
|C-FRN     |fully relational network with re-arrangments and convolutional aggregation at all layers|
|C-RN      |relational network with re-arrangments and convolutional first layer|
|d-DT      |Decision Tree Classifier|
|d-RF      |Random Forest Classifier|
|d-EF      |Extreme Forest Classifier|

#### 2.1 Input data

Initial numbers and their elementwise comparison vector to help as prior knowledge about the existing relations between numbers.

#### 2.2 Comparison

Between decision tree and multilayer perceptron. Multiple variants of decision trees: decision, forest and extreme. The neural network is mostly tuned by exploring different options for activations functions and increasing the number of layers/neurons until there are no visible improvements. 

### 3. Results


| | |
|:-:|:-:|
|Epoch 1.000 - 5 out of 10|Epoch 5.000 - 6 out of 10|
|![l_1000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_1000.png)|![l_5000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_5000.png)|
|Epoch 10.000 - 7 out of 10|Epoch 22.000 - 9 out of 10|
|![l_10000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_10000.png)|![l_22000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_22000.png)|

SAMPLE RESULTS DISPLAY FOR BASELINE MODEL ON AN ARRAY OF 10

#### 4. Metrics

We compute a partial accuracy - the average number of elements whose target position is correctly guessed. Example: we run the experiment on an array of 10 elements and the test dataset is comprised of 3 such arrays with 7, 7, and 9 correct guesses. Then the partial accuracy is (7 + 7 + 9)/3 = 7.7.

Later on, we use a normalized accuracy - the average percentage of correctly guessed elements from the whole array. Example 7 correct guesses in an array of 9 elements has a normalized accuracy of 7 / 9 = 78%.

#### 5. Results by Dataset

Error range: +/- 1%

- Knowledge Prior: Set 1 vs. Set 2 vs. Set 3

**Set 1 (Data: D)**

|Size|Model + Acc|Normalized Acc (vs DC)|Comment  |
|:--:|:---------:|:--------------------:|:-------:|
|6   |NN  6.0    |100% vs 100%          |full both|
|8   |NN  8.0    |100% vs 100%          |full both|
|9   |NN  8.7    | 97% vs 100%          |+ 3% DC  |
|10  |NN  8.8    | 88% vs  95%          |+ 7% DC  |
|11  |NN  6.3    | 57% vs  66%          |+ 9% DC  |
|12  |NN  4.9    | 41% vs  44%          |+ 3% DC  |
|16  |NN  3.4    | 21% vs  23%          |+ 2% DC,   start poor convergence|
|20  |NN  1.0    |  5% vs   7%          |+ 2% DC, +18% with HP, stays same|

- Basic Set
- Hyperparams tuning necessary when too much diffusion

**Set 2 (Data: DC)**

|Size|Worst |Second  |Best    |Comment              |
|:--:|:----:|:------:|:------:|:-------------------:|
|6   |      |E_96 6.0|NN 6.0  |confirm ratio (1000) |
|8   |D  6.0|E_96 7.9|NN 8.0  |confirm ratio (10000)|
|10  |D  4.0|E_96 7.5|NN 9.5  |confirm ratio (60000)|
|12  |      |NN 5.3  |E_96 7.2|                     |
|16  |      |NN 3.6  |E_96 5.4|                     |
|20  |      |NN 1.4  |E_96 4.6|NN 4.6 (4L, 2K N, 150K S)|
|24  |      |NN 1.0  |E_96 3.5| ^ (0.0006 lr, 0.7 drop) |

- Set with additional prior knowledge

**Set 3 (Data: C)**

|Size|Partial Acc|Normal Acc (vs DC)|Comment          |
|:--:|:---------:|:----------------:|:---------------:|
|6   |NN 6.0     |100% vs 100%      |                 |
|8   |NN 7.9     | 99% vs 100%      |+ 1% DC converges|
|9   |NN 8.4     | 93% vs 100%      |+ 7% DC converges|
|10  |NN 8.3     | 83% vs  95%      |+12% DC converges|
|11  |NN 8.1     | 74% vs  66%      |- 8% DC          |
|12  |NN 7.3     | 61% vs  44%      |-17% DC          |
|16  |NN 5.1     | 32% vs  23%      |- 9% DC          |
|20  |NN 3.2     | 16% vs   7%      |- 9% DC fails converge|

- Set with total value abstraction
- So far seems the most scalable (accuracy drops slower)


#### 6. Scalability

We show how this poses scalability problems for the chosen machine learning models (neural networks and decision trees). For instance:

Using DC dataset.

|Size|   6|   8|   9|  10|  11|  12|  16|  20|  24|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|  NN|1.00|1.00|1.00|0.95|0.66|0.44|0.23|0.07|0.04|
|  DT|1.00|0.99|0.92|0.75|0.68|0.60|0.34|0.23|0.15|
|Rand|0.17|0.12|0.11|0.10|0.09|0.08|0.06|0.05|0.04|

Up to N = 10 we get very good results with both models. 
For N = 24, the neural network starts predicting worse than random guessing.
The neural network performs better than the decision tree-based models up to N = 10, then it has a sudden drop. However, decision trees have a slower drop in accuracy.

Using all datasets

|Size|   6|   8|   9|  10|  11|  12|  16|  20|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|   D|1.00|1.00|0.97|0.88|0.57|0.41|0.21|0.05|
|  DC|1.00|1.00|1.00|0.95|0.66|0.44|0.23|0.07|
|   C|1.00|0.99|0.93|0.83|0.74|0.61|0.32|0.16|

|Accuracy|Model Description|
|:------:|:---------------:|
|![asm_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/asm.png)|Extreme forest with 96 estimators vs. multilayer_perceptron with ~1000 neurons in 4 layers (512, 256, 128) dense + N multi-label outputs. Rand ~ 1/N|
|![asd_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/asd.png)|Baseline model, 3 types of feature sets (datasets). Normalized accuracy expressed in [0,1]|
|![ad_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/ad.png)|Accuracy fot architecture based on previous design for different input representations (different datasets). [input, aN^2, bN, output]. In this case, layer widths are in relation to N = 20.|

### 7. Problem Complexity

Next, we try to find some of the underlying reasons. For instance, by measuring the impact of data representation: bare numbers or numbers with an order relation. 

This is done by investigating the properties of the input and target spaces.

- important space dimensions (set cardinality):
- N! vs. MAXINT!/(MAXINT-N)! vs. 2^((N)x(N-1)/2) vs. 2^N

Then we look for changes in the models or the problem formulation that could help improve our solution.

|Growth|Description|
|:----:|:---------:|
|![pc_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/pc.png)|Figure showing the **data growth** on a logarithmic scale. Input: **magenta** - number of valid inputs in relation to N. Output: **red** - number of valid outputs in relation to N.|
||Depending on how the problem is formalized we can manipulate the number of valid inputs with lossless compression (re-abstractization).|
|**Arrangements**|The number of possible arrangements of unique numbers up to MAXINT of length N (INPUT DATA)|
|**Permutations**|The number of possible permutations of length N(OUTPUT and INPUT ORDER RELATIONS)|
|**Exponential**|2 to the power of N - for reference purpose|
|**Combinations**|The number of possible relations between pairs of numbers|


### 8. Design Model

|Figure|Interpretation|
|:----:|:------------:|
|![asbs_10](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/asbs_10.png)|Comparison of the baseline and the design models convergence for N = 10|


**Focus area, take-away message: 99 - 57 = 42% accuracy improvement for size 20**


**Increasing & decreasing the size**

- D0, DC, N10, s   , after finish 	   (1.00 vs 1.00)
- D0, DC, N12, s   , after finish      (1.00 vs 1.00)
- D0, DC, N16, s8.1, after finish 16.0 (1.00 vs 0.77)
- D0, DC, N20, s6.6, after finish 19.9 (0.99 vs 0.57)
- D0, DC, N24, s6.0, after finish 19.1 (0.80 vs 0.40)
- D0, DC, N28  s5.6, after finish 13.8 (0.49 vs 0.30) 
- D0, DC, N30, s5.4, after finish 10.8 (0.36 vs 0.28)

### 9. Relational Model

|Figure|Interpretation|
|:----:|:------------:|
|![bscs_10_20](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/bscs_10_20.png)|Comparison of the design and the relational models convergence for N = 10, 20|
|![bscs_30](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/bscs_30.png)|Comparison of the design and the relational models convergence for N = 30|

**With Relational Net**

- Q: relational net
- R: fully relational net
- net Q, D (Data), N28 (batch 64 = half) s4.4, 6.4, 7.6, 8.7, 9.8, 10.6 after finish 27.2 (0.97)
- net R, D (Data), N30 (batch 64 = half) at finish 20.1 (0.67)
- net R, D (Data), N32 (batch 64 = half) at finish 19.6 (0.61)

The SortNet model can be modified such that it has learnable weights an in principle could learn any total order relation between the array elements. For this reason we re-name it to RelationalNet.

What makes it very applicable is its incorporation of 3 fundamental principles in designing supervised bottom-up learning networks.

These are:

- Re-combination of input chunks (relational unit)
- Aggregation of learned activations (aggregator unit)
- Directed acyclic graph structure (logical reasoning from extracted entities)

It is remarkable how handful convolutions come in this case: they facilitate the writing of learnable aggregations, while allowing shared weights which greatly increase the learning efficiency.


**Focus area, take-away message: near perfect accuracy becomes accessible for size 30**


### 10. SortNet Model

|Figure|Interpretation|
|:----:|:------------:|
|![sort_net_2](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/images/sort_net_2.png)|A fixed weights (hard-coded) neural model which solves the problem with 100% accuracy for up to N = 500 (depending on hardware)|

### 11. Conclusions

In the first part we looked at challenges posed to fairly complex neural networks by the failure to abstract the structure of the data from a relational point of view. We overcome this obstacle by re-abstracting the data and by creating a model designed to exploit the relational structure. This way we were able to scale from high accuracy for 10 elements to high accuracy for 20 elements (**Sections 3-8**). 

Moreover, we presented an algorithm for sorting that is written in "neural networks" language, which a model with the capacity to represent it; could learn the weights to implement it. We compare the differences between a baseline knowledge-agnostic model, with a well-designed model and the pseudo-code (hard-coded) model - which was tried experimentally and scales well up to 500 elements (**Section 10**).

Afterwards, we designed an adaptive model capable of learning these weights, which was able to scale its ability to learn how to sort a collection of objects up to 30 elements (**Section 9**).

#### 11.1 Summary of Results

#### 11.2 Interpretation of Results

### 12. Future Work

### Bibliography

#### Related work

- Sorting-related experiments with deep learning [A], [B].
- Relational reasoning [C], [D].
- Semantic Relations in sequential representations (word embeddings) [E], [F].
- Convolutional Neural Networks for Natural Language Processing [G], [H], [I].
- Links to information theory and deep learning [X], [Y].
- Bottleneck principle [Z], [W].

There are some other works where simple algorithms are inferred via neural networks. For example, in [A], the operations of copying and sorting an array are performed with a fully differentiable network connected to an eternal memory via attention.

In another approach, [B] presents a sorting experiment for sets of numbers using a sequence to sequence model. The fundamental question which the study raises is how to represent sets as a sequence when the underlying order is not known or doesn't exist.

The importance of prior information and pre-learned intermediate concepts. Composition of 2 highly non-linear tasks and other hypothesis such as local minima obstacle and guided/transfer learning [X].

Gradients in highly composed functions or hard constraints. Accuracy as a function of input dimensionality. Gradient based methods cannot learn reasonably fast random parities and linear-periodic functions [Y].

#### Core deep learning models

[A] *Neural Turing Machines*, A. Graves, G. Wayne, and I. Danihelka, arXiv:1410.5401v2, 2014.

[B] *OrderMatters: Sequence to Sequence for Sets*, O. Vinyals, S. Bengio, and M. Kudlur, in 4th International Conference on Learning Representations (ICLR), 2016.

[C] *A simple neural network module for relational reasoning*, A. Santoro et al, 2017

[D] *Compositional Attention Networks for Machine Reasoning*, D. Hudson and C. Manning, 2018

[E] *Distributed Representations of Words and Phrases and their Compositionally*, T. Mikolov, 2013

[F] *Distributed Representations of Sentences and Documents*, Q. Le, T. Mikolov, 2014

[G] *Convolutional Neural Networks for Sentence Classification*, Y. Kim, 2014

[H] *Convolutional Neural Networks for Modelling Sentences*, N. Kalchbrenner,
2014

[I] *Deep Learning applied to Natural Language Processing*, M-M. Lopez, J.
Kalita, 2017

#### Deep learning theory

[X] *Knowledge Matters: Importance of Prior Information for Optimization*, C. Gulcehre and Y. Bengio, Journal of Machine Learning Research, 2016.

[Y] *Failures of Gradient-Based Deep Learning*, S. Shalev-Shwartz and O. Shamir and S. Shammah, arXiv:1703.07950, 2017.

[Z] *Opening the black box of Deep Neural Networks via Information*, Ravid Schwartz-Ziv and Naftali Tishby, 2017

[W] *Deep Learning and Information Bottleneck Principle*, Naftali Tishby and Noga Zaslavsky, 2015

#### Co-authored work on which the experiment builds upon (published/unpublished)

Tags: Algorithms; Machine Learning; Program Induction; Cognitive side of Programming; Hierarchical Segmentation;

[1] *Finding Patterns in Visualization of Programs*, conference paper, PPIG 2017

[2] *Using the Self-Organizing Map to cluster Computer Programs based on Spatio-Temporal Features extracted from Execution Traces*, unpublished, 4 versions presented for Masters & Research reports, 2017-2018 

[3] *Detection and Emulation of Algorithms from Execution Traces*, under review KNOSYS Elsevier journal, 2017-2018, submitted Feb 2019

[4] *Hierarchical segmentation of graphical interfaces for Document Object Model reconstruction*, workshop short paper & poster, ICML 2018

[5] *Modeling cognitive processes underlying computer programming*, conference paper, PPIG 2018

[6] *License Plates Detection in Street View Images*, unpublished, Accesa technical report 2018

#### Misc

[J] *Critique of Pure Reason*, I. Kant

[K] *Mathematics, Form and Function*, S. MacLane

[L] *The Bitter Lesson*, R. Sutton

[M] *DAG*, Matlab, url = https://www.mathworks.com/help/deeplearning/ref/dagnetwork.html

[N] *CNN*, Wikipedia, url = https://en.wikipedia.org/wiki/Convolutional_neural_network

[O] *Word2Vec*, Tensorflow, url = https://www.tensorflow.org/tutorials/word2vec

[P] *Problema SortNet*, M. Patrascu, Infoarena, url = https://infoarena.ro/problema/sortnet 

[Q]

[R]

[S] 

[T]

[U]

[V]