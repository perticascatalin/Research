# MastersExperiments

Thesis Title: **Compositionality of relational operations in neural networks**

## E1: Order relations compositionality in a sequence of numbers (mostly done)

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

#### Concepts

**Concepts list:**

- information bottleneck (intro+ formula+ impl- debate-)
- scalability & problem complexity (intro+ formula+ impl+ debate-)
- learning complexity & separability (approached)
- compositionality - multiple learners (approached, experiments+)
- relation to mathematical sets (dk)
- sequences (dk)

**Principles of writing:**

- keep theory & speculation in balance with maths and experimental part
- link neighboring points through story and insight
- keep flat structure, so that it is clear where to add certain stuff
- polish partial parts, but leave way for new parts to be created

**Applications:**

- computer vision
- all problems involving permutations (even NP problems)
- rawest example: sequence of inputs, reorder inputs such that some function of the inputs in that particular order is maximized
- the number of possible permutations grows very fast with N and the possible number of order relations even faster.

#### Information Bottleneck

The information bottleneck principle links the analysis of deep neural networks to the domain of information theory. The key ideas of the principle are introduced as mathematical tools in a series of papers ([Z], [W]). These are: measuring the information propagation level against the theoretical upper bound and exploring the variation in information propagation during learning (the accuracy gain phase and the compression phase). How information is compressed inside a neural network depends on the data representation which can carry information about target outputs at different relevance levels. In practice, we can estimate the mutual information between random variables representing the input, the activation at a certain level and the output of the network. We can use these measures to intuitively analyze what is going through the network and then find out the factors which contribute to the success of the model design and parametrization.

### 2. Models

Briefly describe the models used and the intuitions behind them. Illustrate their structure and mechanism.

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

#### 4. Metrics

We compute a partial accuracy - the average number of elements whose target position is correctly guessed. Example: we run the experiment on an array of 10 elements and the test dataset is comprised of 3 such arrays with 7, 7, and 9 correct guesses. Then the partial accuracy is (7 + 7 + 9)/3 = 7.7.

Later on, we use a normalized accuracy - the average percentage of correctly guessed elements from the whole array. Example 7 correct guesses in an array of 9 elements has a normalized accuracy of 7 / 9 = 78%.

#### 5. Results by Dataset

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
|20  |      |NN 1.4  |E_96 4.6|NN 4.6 (4L, 2K N, 150K S)|
|24  |      |NN 1.0  |E_96 3.5| ^ (0.0006 lr, 0.7 drop) |

- Set with additional prior knowledge

**Set 3**

- Data: C

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

**START NOT MOVED**

**Exploration of [400,200] layering**

- With HP Range 20 (60.000) : NN 8.6 (0.43 vs 0.23) [400,200] rest same, improv 20%
- With HP Range 20 (150.000) add data: NN 8.9 (0.45 vs 0.43) 2% very marginal improvement

**Range 20**

- D6 Less dropout (0.6), no additional data: NN 10.3 (0.52 vs 0.43) improv 9%
- D4 Less dropout (0.4), same: NN prev s6.1, now s6.4, after finish 12.1 (0.61 vs 0.43) improv 18%
- D2 Even less (0.2), same: NN now s6.5, after finish 14.7 (0.74 vs 0.43) improv 31%
- D0 None (0.0), same: NN now s6.6, after finish 19.9 (0.99 vs 0.43) improv 56%

**Comparison to baseline data**

D2 Same, but back to data NN now s5.1, after finish 8.7 (0.44 vs 0.74)
D0 NN now s5.3, after finish 11.4 (0.57 vs 0.99)

**======================**

**Focus area, take-away message: 99 - 57 = 42% accuracy improvement for size 20**

**======================**

**Increasing & decreasing the size**

- D0, DC, N10, s   , after finish 	   (1.00 vs 1.00)
- D0, DC, N12, s   , after finish      (1.00 vs 1.00)
- D0, DC, N16, s8.1, after finish 16.0 (1.00 vs 0.77)
- D0, DC, N20, s6.6, after finish 19.9 (0.99 vs 0.57)
- D0, DC, N24, s6.0, after finish 19.1 (0.80 vs 0.40)
- D0, DC, N28  s5.6, after finish 13.8 (0.49 vs 0.30) 
- D0, DC, N30, s5.4, after finish 10.8 (0.36 vs 0.28)

**With Relational Net**

- Q: relational net
- R: fully relational net
- net Q, D (Data), N28 (batch 64 = half) s4.4, 6.4, 7.6, 8.7, 9.8, 10.6 after finish 27.2 (0.97)
- net R, D (Data), N32 (batch 64 = half, only 10000 samples = 1/6) at finish 19.6 (0.61)

The SortNet model can be modified such that it has learnable weights an in principle could learn any total order relation between the array elements. For this reason we re-name it to RelationalNet.

What makes it very applicable is its incorporation of 3 fundamental principles in designing supervised bottom-up learning networks.

These are:

- Re-combination of input chunks (relational unit)
- Aggregation of learned activations (aggregator unit)
- Directed acyclic graph structure (logical reasoning from extracted entities)

It is remarkable how handful convolutions come in this case: they facilitate the writing of learnable aggregations, while allowing shared weights which greatly increase the learning efficiency.

**======================**

**Focus area, take-away message: near perfect accuracy becomes accessible for size 30**

**======================**

**Exploration of A.[800,200], B.[800,400]**

- A. D0, DC, N24, s5.9, after finish 21.5 (0.90)
- B. D0, DC, N24, s6.9, after finish 10.6 (0.44)
- B is a lot worse, probably needs dropout or HP change
- continue with A

**Flat architecture**

- F_30 [1000] 9.0
- F_20 [400] 19.3, d 8.3
- G_30 [30] 7.4
- G_20 [20] 9.0, d 3.1

- Diffusion and design vs. dropout and information bottleneck

**STOP NOT MOVED**

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


Scalability vs. problem complexity views:

- **scalability - done**
- **problem complexity - done**

|Accuracy|Model Description|
|:------:|:---------------:|
|![asm_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/asm.png)|Extreme forest with 96 estimators vs. multilayer_perceptron with ~1000 neurons in 4 layers (512, 256, 128) dense + N multi-label outputs. Rand ~ 1/N|
|![asd_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/asd.png)|Add description|
|![ad_plot](https://raw.githubusercontent.com/perticascatalin/MastersExperiments/master/Permutation/results/ad.png)|Add description|

### 7. Problem Complexity

Next, we try to find some of the underlying reasons. For instance, by measuring the impact of data representation: bare numbers or numbers with an order relation. This is done by investigating the properties of the input and target spaces.

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