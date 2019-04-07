# MastersExperiments

Thesis Title: **Scaled statistical learning on compositions of functions for relational operations**

## E1: Order relations compositionality in a sequence of numbers (in progress)

Applications:

- computer vision
- all problems involving permutations (even NP problems)
- rawest example: sequence of inputs, reorder inputs such that some function of the inputs in that particular order is maximized
- the number of possible permutations grows very fast with N and the possible number of order relations even faster.

### 1. Introduction

- compositionality (approached)
- information bottleneck (intro+ formula- impl- debate-)
- scalability, separability - multiple learners (approached)
- learning complexity (approached)
- relation to mathematical sets (dk)
- sequences (dk)

#### Information Bottleneck

The information bottleneck principle refers to the link between information theory and deep neural networks. How information is compressed inside a neural network depends on the data representation which can carry information about target outputs at different relevance levels. In practice, we can estimate the mutual information between random variables representing the input, the activation at a certain level and the output of the network. We can use these measures to intuitively analyze what is going through the network and then find out the factors which contribute to the success of the model parametrization and design.

#### 1.1 Relevance and priority (MOVED Related Work)

Whatever tasks they have to model, neural networks often encounter the problem of modeling relevance and priority - how to assign 1st, 2nd, 3rd and so on to data chunks such that it answers some questions. Such an example is the spatial ordering of elements in a visual scene as shown below.

|Example|Description|
|:-----:|:---------:|
|![logico_visual](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/images/logico_visual.png)|Start from logico-visual tasks learned by a neural network and continue with set theory and an analysis to solve particular questions.|

Understanding how neural networks and other supervised learning models learn to answer such questions by processing large amounts of labeled data is of importance to any visual automated tasks from tracking objects of interest in scenes to self-driving cars. Previous work on how to model this learning task includes [C] and [D]. The two papers are conceptually different, but both of them use CLEVR dataset as benchmark. [C] presents a neural network module which is claimed to better model relational reasoning. [D] goes into the area of attention, memory and composition. However, what they have in common is the apparently hard task of learning composed functions. 

#### 1.2 Visual Tasks (MOVED Related Work)

For instace, when asking a question about the nearest object to the black sphere, a neural network model has to learn the function of identifying the blue sphere, take the result, look for the furthest object to it, then retrieve it and tell something about its properties, which is yet another task to learn. One may conclude that answering such a simple question is a task in 3 steps, or is a task composed of 3 sub-tasks.

Neural networks owe some of their success to the ability of learning higher level compositions. From this perspective, we look at the challenges that come with an increased number of objects (above 10 typically). Some of our experiments might be able to empirically prove that NNs do not learn the correct compositions all the time, especially when the number of conceptually relevant objects gets higher than ~10. However, this is something that can be overcome at least partially through relational representation. The task of sorting an array of numbers (size N) is very expressive for this purpose - although the end result has to satisfy the constraint that all numbers are in their correct place, the composition is not N-fold, but 2-fold instead (at any time we only need to know how to compare 2 elements).

|Example|Description|
|:-----:|:---------:|
|![relational_sequence_sets](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/images/relational_sequence_sets.png)|Data relations: a sequence representing a certain order of the objects - set w. order relation. The set of all permutations. Symmetric group (composition of f and g to visit whole group).|

#### 1.3 Mathematical Structures (MOVED Order Relations)

In this study we run experiments on data with mathematical properties such as sets, permutations and groups. And then we look at the impact of their solving difficulty in practical applications. Along the way we mention previously encountered challenges in machine learning from which the problem at hand suffers, but the focus will be on how we can exploit its structure.

The 0-1 hypothesis in permutation representation states that any permutation can be generate through an f-g path of transformations from the identical permutation.

The special arrangement hypothesis assumes an arrangement of multiple neural networks specialized in recreating a specific subset of transformations such that an input can be forwarded through a path or a cycle such that a specific state is always achieved.

#### 1.4 Predicting the correct order (MOVED Sequence of Numbers)

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

#### 1.5 Order Relations (MOVED Sequence of Numbers)

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

#### 1.6 Harder Stuff

Looking at the input in a sequential manner or deriving rules for correct input parsing - sorted numbers should be easily mapped to 0, 1, 2, 3, ... what if many numbers from a random permutation of numbers? Distributive attention - how to paralelly process 2 different data streams.

Number of total inputs for a given permutation:

How many input states are represented by a node.

Assign first p0 such that you can keep assigning p1,p2... (N-1) avail
MAXINT = 10
max_assign = 9 0 up to 9-N

Assign second p1 such s.t. p2, p3

Learning specific permutations and the mapping within them through a Cayley Graph. Distributing the types of inputs to different models goes in the direction of AutoML.

### 2. Models

Briefly describe the models used and the intuitions behind them. Illustrate their structure and mechanism.

#### 2.1 Input data

Initial numbers and their elementwise comparison vector to help as prior knowledge about the existing relations between numbers.

#### 2.2 Comparison

Between decision tree and multilayer perceptron. Multiple variants of decision trees: decision, forest and extreme. The neural network is mostly tuned by exploring different options for activations functions and increasing the number of layers/neurons until there are no visible improvements. 

### 3. Results

#### 3.1 Views

Display graphic array before and after: 

- **barchart-done**
- stacking
- coloring

| | |
|:-:|:-:|
|Epoch 1.000 - 5 out of 10|Epoch 5.000 - 6 out of 10|
|![l_1000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_1000.png)|![l_5000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_5000.png)|
|Epoch 10.000 - 7 out of 10|Epoch 22.000 - 9 out of 10|
|![l_10000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_10000.png)|![l_22000](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/labels/labels_22000.png)|


#### 3.2 Overall View

Where does the model fail?
Split into categories [useless (less than 10% acc) - guess (max 5) - good guess (7) - problem solved (N)]. Rank all the samples in a batch - by accuracy or mean squared error (proximity).

#### 3.3 Metrics (MOVED Implementation and Experiments)

We compute a partial accuracy - the average number of elements whose target position is correctly guessed. Example: we run the experiment on an array of 10 elements and the test dataset is comprised of 3 such arrays with 7, 7, and 9 correct guesses. Then the partial accuracy is (7 + 7 + 9)/3 = 7.7.


Error range: +/- 1%

- Knowledge Prior: Set 1 vs. Set 2 vs. Set 3

**Set 1**

- Data: D

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
- Min 1 element guessed threshold stop investigation
- Hyperparams tuning necessary when too much diffusion

**Set 2**

- Data: DC

|Size|Worst |Second  |Best    |Comment              |
|:--:|:----:|:------:|:------:|:-------------------:|
|6   |      |E_96 6.0|NN 6.0  |confirm ratio (1000) |
|8   |D  6.0|E_96 7.9|NN 8.0  |confirm ratio (10000)|
|10  |D  4.0|E_96 7.5|NN 9.5  |confirm ratio (60000)|
|12  |      |NN 5.3  |E_96 7.2|                     |
|16  |      |NN 3.6  |E_96 5.4|                     |
|20  |      |NN 1.4  |E_96 4.6|NN nc 4.6 (4 layers, 2000 neurons, 150000 samples, 0.0006 lr, 0.7 dropout)|
|24  |      |NN 1.0  |E_96 3.5| |

- Set with additional prior knowledge

**Set 3**

- Data: C
- Range 6:					NN 6.0 full both - not necessary
- Range 8:					NN 7.9 (0.99 vs 1.00) improv 1% DC converges
- Range 9:					NN 8.4 (0.93 vs 1.00) improv 7% DC converges
- Range 10:					NN 8.3 (0.83 vs 0.95) improv 12% DC converges
- Range 11:					NN 8.1 (0.74 vs 0.66) decrev 8% DC
- Range 12:					NN 7.3 (0.61 vs 0.44) decrev 17% DC
- Range 16:					NN 5.1 (0.32 vs 0.23) decrev 9% DC
- Range 20:					NN 3.2 (0.16 vs 0.07 vs 0.23) fails converge, decrev 9% DC

[400,200]

With HP Range 20 (60.000) : NN 8.6 (0.43 vs 0.23) [400,200] rest same, improv 20%
With HP Range 20 (150.000) add data: NN 8.9 (0.45 vs 0.43) 2% very marginal improvement

Range 20
D6 Less dropout (0.6), no additional data: NN 10.3 (0.52 vs 0.43) improv 9%
D4 Less dropout (0.4), same: NN prev s6.1, now s6.4, after finish 12.1 (0.61 vs 0.43) improv 18%
D2 Even less (0.2), same: NN now s6.5, after finish 14.7 (0.74 vs 0.43) improv 31%
D0 None (0.0), same: NN now s6.6, after finish 19.9 (0.99 vs 0.43) improv 56%

Comparison to baseline data
D0 Same, but back to data: NN now s5.3, after finish 11.4 (0.57 vs )
D2 NN now s5.1, after finish 8.7 (0.44 vs )

D0, DC, N24, s6.0, after finish 19.1 (0.80 vs )
D0, DC, N30, s5.4, after finish 10.8 (0.36 vs )

- Set with total value abstraction
- So far seems the most scalable (accuracy drops slower)
- Diffusion and design vs. dropout and information bottleneck

#### 3.4 Scalability (MOVED Implementation and Experiments)

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


Scalability vs. problem complexity views:

- **scalability - done**
- problem complexity
- N factorial vs. number of samples ratio threshold

|Accuracy|Model Description|
|:------:|:---------------:|
|![asm_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/asm.png)|Extreme forest with 96 estimators vs. multilayer_perceptron with ~1000 neurons in 4 layers (512, 256, 128) dense + N multi-label outputs. Rand ~ 1/N|
|![asd_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/asd.png)|Add description|

### 4. Problem Complexity

Next, we try to find some of the underlying reasons. For instance, by measuring the impact of data representation: bare numbers or numbers with an order relation. This is done by investigating the properties of the input and target spaces.

- important space dimensions (set cardinality):
- N! vs. MAXINT!/(MAXINT-N)! vs. 2^((N)x(N-1)/2) vs. 2^N

Then we look for changes in the models or the problem formulation that could help improve our solution.

### 5. In-progress Work

*To investigate further*

1. Representation: sequence of numbers vs. bag of numbers (count sort)

2. Multilabel Classification.
Vectors for small arrays of numbers. 
Largest enhancements and smallest shrinking preserve array class (all labels).
Bent surface on n-2 space.

3. Regression.
Directly predict the numbers in the correct order.
Neural network has to store the numbers.

4. Temporal Generation. Sequence to sequence. Results which can be compared to previous papers on read-write NN.

5. Pre-order to/vs. Post-order

6. Constraint non-identical labels

7. Input-output matching cardinality

8. Feature importances from decision tree

9. Trick network by having some fixed positions.
K out of N elements are fixed such that solving the problem for them yields better results than for bothering with the rest of the elements.

10. Minimum number of samples to reach decent accuracy.

### 6. Experiments vs Theory

1. Adding comparison prior knowledge to data

2. Storage capability: predicting the order vs. predicting elements in order

3. Maximizing expected value vs. optimization

4. Impurity vs. local minima

5. Attribute split vs. non-linear combinations

6. Models scalability - DONE

7. Problem Complexity

### 7. Related work

Links to information theory and deep learning [X], [Y].
Sorting-related experiments with deep learning [A], [B].
Bottleneck principle [Z], [W].

The importance of prior information and pre-learned intermediate concepts. Composition of 2 highly non-linear tasks and other hypothesis such as local minima obstacle and guided/transfer learning [X].

Gradients in highly composed functions or hard constraints. Accuracy as a function of input dimensionality. Gradient based methods cannot learn reasonably fast random parities and linear-periodic functions [Y].

There are some other works where simple algorithms are inferred via neural networks. For example, in [A], the operations of copying and sorting an array are performed with a fully differentiable network connected to an eternal memory via attention. (**MOVED**)

In another approach, [B] presents a sorting experiment for sets of numbers using a sequence to sequence model. The fundamental question which the study raises is how to represent sets as a sequence when the underlying order is not known or doesn't exist. (**MOVED**)

### 8. References

[A] A. Graves, G. Wayne, and I. Danihelka, “Neural Turing Machines,” arXiv:1410.5401v2, 2014.

[B] O. Vinyals, S. Bengio, and M. Kudlur, “OrderMatters: Sequence to Sequence for Sets”, in 4th International Conference on Learning Representations (ICLR), 2016.

[C] A. Santoro et al, "A simple neural network module for relational reasoning"

[D] D. Hudson and C. Manning, "Compositional Attention Networks for Machine Reasoning"

[X] C. Gulcehre and Y. Bengio, "Knowledge Matters: Importance of Prior Information for Optimization", Journal of Machine Learning Research, 2016.

[Y] S. Shalev-Shwartz and O. Shamir and S. Shammah, "Failures of Gradient-Based Deep Learning", arXiv:1703.07950, 2017.

[Z] Ravid Schwartz-Ziv and Naftali Tishby, "Opening the black box of Deep Neural Networks via Information"

[W] Naftali Tishby and Noga Zaslavsky "Deep Learning and Information Bottleneck Principle"