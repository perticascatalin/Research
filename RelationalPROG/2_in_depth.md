# Relational reasoning in deep learning: a parallel between solving visual and programming tasks

## Report: Studies on the initial Bibliography (1-6)

This report continues the study started in the [thesis proposal](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/1_exec_abstract.md) by developing an in-depth analysis of the selected bibliography. The review on related literature is carried out with regards to our previously set research goals.

**Overall Goals:**

- Research possible ways to integrate relational reasoning in deep learning models
- Investigate the role of relational reasoning in solving programming tasks

**Content**

**4. Extraction of Visual Relationships**:

[4.1. A Simple Neural Network Model for Relational Reasoning](https://arxiv.org/pdf/1706.01427.pdf)

[4.2. Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf)

[4.3. Recurrent Relational Networks](https://arxiv.org/pdf/1711.08028v4.pdf)

**5. Neural Problem Solving, Program Synthesis, Source Code**

Multiple tasks: input-output pairs

Neural Problem Solving: Latent Program Induction

[5.1 On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547.pdf)

DSL based, Program Synthesis

[5.2 Deep Coder: Learning to Write Programs](https://arxiv.org/pdf/1611.01989.pdf)

[5.3 Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/pdf/2006.08381.pdf)

**5.4 Source Code Classification and detection**: Abstract Syntax Tree based Neural Networks

[5.4.1 A Novel Neural Source Code Representation based on Abstract Syntax Tree](http://xuwang.tech/paper/astnn_icse2019.pdf)

[5.4.2 Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf)

**5.5 Directions in Program Induction**

## 4. Extraction of Visual Relationships

### 4.1 A simple Neural Network Model for Relational Reasoning

This study presents a general machine learning model used for solving relational tasks. The relational network (RN) model contains a built-in mechanism for capturing core common properties of relational reasoning. From this perspective, we can draw parallels to other machine learning models designed with different properties in mind, such as the well known:

- CNNs used for extracting spatial and translation invariant properties
- LSTMs used for extracting sequential dependencies

The RN is used on 3 different tasks, all of which require some kind of relational inference on a set of objects:

- Visual question answering (datasets: CLEVR - 3d, Sort-of-CLEVR - 2d)
- Text-based question answering (dataset: bAbi)
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

|Sort-of-CLEVR|CLEVR|
|:-----:|:---------:|
|![Sort-of-CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/sort_of_clevr.png)|![CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/clevr.png)|
|What is the color of the nearest object to the yellow object?| What size is the cylinder that is left of the brown metal thing that is left of the big sphere?|
|Green|Small|

The second step is to examine how the make the object extraction question dependent. Questions are processed word-by-word (list of integers assigned to words) by an LSTM and the final state (an embedding of the question) is passed on to the RN together with the objects pair.

These triples are batched through an MLP for learning the relational function g (the output of which is a measure of relevance for object pairs). These outputs are then aggregated by a second MLP which learns the function f used for providing the answer to the question.

Finally, the information about what constitues objects and how to parse the question is backpropagated in the CNN and in the LSTM. This approach shows a 27% improvement in accuracy on the CLEVR  dataset compared to the state-of-the-art at the time of the publication and a 31% improvement in accuracy on the Sort-of-CLEVR dataset compared to a baseline CNN-MLP architecture.

Sample open source [code](https://github.com/clvrai/Relation-Network-Tensorflow) applying RN to the Sort-of-CLEVR dataset.

Code running logs: see [practice report](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/5_practice_report.md)

### 4.2 Compositional Attention Networks for Machine Reasoning

The study presents a recurrent neural network architecture which relies on structural constraints in order to guide the network towards compositional reasoning and to facilitate interpretability of the inferred results. The network is called MAC (memory, attention, composition) because of its specific architecture which separates control from memory and imposes structural constraints that regulate interaction. As opposed to the RN, which is a module, MAC is a network learning cell.

This design is based on more recent advances in machine learning, where a trend to adopt symbolic structures (resembling expression trees of programming languages) can be noticed. However, these systems generally rely on externally provided structured representations. The MAC architecture addresses this issue in a more general fashion, by decomposing a problem into a series of attention-based reasoning steps. These steps aggregate relevant information from the memory and knowledge base and then updates the memory accordingly. This model is applied on the CLEVR dataset, which has a very suitable structure for emphasizing MAC's ability to learn step-by-step reasoning. 

Components:

- Control unit
- Read unit
- Write unit
- Control state
- Memory state
- Knowledge base

|MAC Network|MAC Cell|
|:---------:|:------:|
|![Network](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/mac.png)|![Cell](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/mac_cell.png)|

Results: 98.9% accuracy on CLEVR compared to 95.5% accuracy obtained by the earlier relational network. An additional claimed advantage is that is requires (5x) less training data than other models to achieve strong results.

Other memory based approaches: (A. Graves et al. [12]). Example applications: graph problems - finding shortest paths. A detailed overview of machine learning approaches (graph neural networks) to solving combinatorial optimization problems can be found in (Q. Cappart et al. [13]).

Original open source [code](https://github.com/stanfordnlp/mac-network) implementation of MAC network.

Code running logs: see [practice report](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/5_practice_report.md)

### 4.3 Recurrent Relational Networks

Learning to solve tasks that require a chain of interdependent steps of relational inference, such as:

- answering complex questions about the relationships between objects
- solving puzzles where the smaller elements of a solution mutually constrain each other

Introduces the recurrent relational network (RRN), which:

- operates on a graph representation of objects
- is a generalization of the relational network RN
- solves 20/20 bAbi tasks, compared to 18/20 for RN

Introduces a new dataset Pretty-CLEVR, where the number of relational reasoning steps that are required to obtain the answer can be controlled.

Comparison against RN on the Pretty-CLEVR dataset and on Sudoku puzzle solving.

![Pretty-CLEVR comparison](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/pretty_clevr.png)

## 5. Neural Problem Solving, Program Synthesis, Source Code

### 5.1 On the Measure of Intelligence

This paper discusses approaches to define intelligence and makes a case for better definitions and benchmark datasets. The main points are:

- the need to define & evaluate intelligence (capacity for reasoning) in a way that enables comparisons between 2 systems
- 2 broad historical conceptions:
	- AI: comparison of skills exhibited by AIs & humans at specific tasks
	- Psychology: leverage modern insight into developmental cognitive psychology; the ability to acquire new skills on previously unseen tasks
- since skills are highly modulated by prior knowledge & experience, then unlimited priors & training data allow experiments to buy aribitrary levels of skills for a system
- proposed definition of intelligence: skill-acquisition efficiency
- difference between measuring skill vs. broad abilities
- leverage multi-task benchmarks as a way to assess robustness and flexibility
- system developer embedding the right abstraction into the system, thus partially solving the problem for the "intelligent agent"
- highlight on concepts (critical pieces in intelligent systems) such as: scope, generalization difficulty, priors & experience

Overall, the study offers a more phylosophical, but grounded in research view on intelligence. It also makes a very good case for the design of general AI benchmark datasets and proposes the ARC dataset. 

|||
|:---:|:---:|
|![Sample 1](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/arc_1.png)|![Sample 2](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/arc_2.png)|
|![Sample 3](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/arc_3.png)|![Sample 4](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/arc_4.png)|

![General Intelligence to Specific Tasks](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/moi.png)

Probably contrary to the author's expectations, the solutions to fitting the ARC dataset hosted as a contest on Kaggle seemd to be very task specific. Many participants divided the tasks into subsets using the same abstractions and created DSLs for solving those, or used different specific methods on subsets of similar tasks. So it would seem that in practice, the most available method is still to human engineer a proper model based on the dataset or to  "manually extract" relevant features. Although this certainly takes much more time for a multi task dataset, the dataset can still be "hacked". However, what is really nice about this dataset is how many different problem solving concepts can be integrated into a simple grid-based input-output set of tasks.

### 5.2 Deep Coder: Learning to Write Programs

#### High-Level Summary

This paper presents an approach to program induction involving the use of neural networks to predict the probability that certain methods (from a predefined DSL) appear in a program satisfying a set of given input-output constraints. These probabilities are then used to optimize the search for the program satisfying the input-output constraints.

Overall, the study showcases a potential solution to the IPS problem (Inductive Program Synthesis), called LIPS (Learning Inductive Program Synthesis). This approach can be split into 4 sub-parts:

- **1. DSL specifications and Program attributes**: the use of a DSL is generally necessary for the purpose of restricting the search space of programs (by abstracting away the technical details). Current program synthesis methods do not work well on general programming languages because of the combinatorial explosion of the search space. However, experiments have been performed on Python ASTs, such as (P. Yin & G. Neubig [10]), where a neural network is used to encode natural language specifications into a latent representation and then a second neural network decodes the representation into an AST. In the case of Deep Coder, the program attributes are represented by the estimated probabilities of a method from the DSL to appear in the target program. These attributes are then further used to reduce the search space.

- **2. Data generation**: the advantages of using DSLs are not limited to search space reduction. One can leverage DSLs to generate synthetic data which can be used for training. In the case of Deep Coder, the generated data consists of small programs which can be evaluated on random input data for the purpose of obtaining input-output pairs. The IO pairs and the program are then used in training a neural network to predict the program attributes used for narrowing down the search problem.

- **3. Machine Learning Model to predict Program Attributes**: M input-output examples (M = 5) are used as input for the network (these are padded with a special value in order to have a fixed length). Next, the encoder part of the network concatenates the input & output examples types, as well as their embeddings into a final program state representation vector. The different examples available are aggregated by averaging. Finally, the network outputs predictions for program attributes (probability that a function from the DSL will appear or will be absent in the program).

|Component|Illustration|
|:-------:|:----------:|
|Sample Program|![Program Example](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/prog.png)|
|MLP Architecture|![Model](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/dcnn.png)|
|MLP Output|![Program Attributes](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attrib.png)|

- **4. Search Strategy guided by estimated Program Attributes**: the study tests and compares 2 different search strategies (DFS, Sort & Add enumeration - a branch and bound type of strategy, Beam Search - a version of BFS), with and without guidance from the estimated program attributes. The approach based on program attributes shows considerable search performance improvements. The comparisons to other baseline methods - Beam Search & the SMT solver from (A. Solar-Lezama [11]) are made available. To note however that the program sizes are quite small, up to T = 5 instructions.

|Search Tree|Search Times|
|:-------:|:----------:|
|![Search](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/g_search.png)|![Times](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/dc_times.png)|

#### Programs Representation

This section summarizes the scope of the DSL used in Deep Coder and briefly explains the complexity of using ASTs, which have the capability to represent general purpose programs and do not have to be synthetically generated, such as DSLs. They could be extracted from open source code instead.

**Domain Specific Language**

The DSL defined in this study is comprised of first order and higher order functions. The supported data types are int, bool and array. Each function is defined on an input type to an output type. Additional methods are defined for equality, comparisons and operations with a predetermined set of numbers. The experiments reported are performed on programs of length up to 5. The DSL is designed to be applicable to simple programming competitions problems.

*First Order Functions*

Head, Last, Take, Drop, ...

*Higher Order Functions*

Map, Filter, Count, Zip, ...

**Abstract Syntax Trees**

In order to understand why ASTs are a challenging program representation in the context of machine learning, we can look at two different implementations for the problem of finding the greatest common divisor. One solution implements Euclid's algorithm, the other iteratively subtracts the smaller number from the larger one until the two are equal.

|Euclid's Algorithm|Iteration|
|:-------:|:----------:|
|![7](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/7.png)|![9](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/9.png)|

| Implementation   | # Lines | # Nodes | # Attributes |
|:----------------:|:-------:|:-------:|:------------:|
| Euclid Algorithm | 7       | 48      | 70           |
| Iteration        | 9       | 73      | 99           |

We can notice that although only a few lines long, both programs require a significantly large number of nodes and attributes when represented as an AST. The role of a DSL in such cases is to condense the representation to a bare minimum by abstracting away technical details.

|Component|Illustration|
|:-------:|:----------:|
|Abstract Syntax Tree|![AST](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/ast.png)|

Sample open source [code1](https://github.com/dkamm/deepcoder), [code2](https://github.com/HiroakiMikami/deep-coder) implementing Deep Coder.

### 5.3 Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning

- a system that learns to solve problems by writing programs
- start by using primitives to learn programs based on input-output pairs
- adds up new symbolic abstractions (refactoring) based on imagined and replayed problems
- learns libraries of concepts
- builds expertise by creating programming languages for expressing domain concepts

Tasks:

- classic inductive programming tasks
- drawing pictures, building scenes
- rediscovers basics of modern functional programming, vector algebra, classical physics

## 5.4 Source Code Classification and Detection (AST based)

### 5.4.1  A Novel Neural Source Code Representation based on Abstract Syntax Tree

- treating code as natural language texts fails to capture syntactical and semantic information
- more long term dependencies are required in modelling source code
- tasks: code clone detection and source code classification

### 5.4.2 Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree

- difference between syntactic and semantic clones
- flow augmented abstract syntax trees, data flow
- datasets: google code jam and big clone bench
- graph neural networks

## 5.5 Directions in Program Induction

### 5.5.1 Latent Induction vs Code Synthesis

The first important bifurcation in the approaches for generating programs is the choice of program representation. When a neural network learns to map input to output, thus solving a programming task, the program is stored in the network and executed by the network through neural activations. This is called latent program induction, because the representation of the generated program is not human-readable.

The second choice is to formulate the problem such that the neural network outputs a program in a language, which is then executed to get the desired output. This is generally referred to as program synthesis.

A comparison of the two approaches applied on string transformation problems is carried in Devlin et al. [21].

Latent programs are written in the language of neural networks, whereas synthesized programs are written in a language of choice. Both approaches have shown success, however it is not possible to pick one that works best because they have different strengths. For instance, induction is more likely to provide a good approximation of the output function for the type of inputs provided, but might not generalize so well for new inputs. On the other hand, synthesis will either find the correct program and generalize the solution well for all inputs, or find the wrong solution which over-fits the presented input. Synthesis is thus more capable, but also the riskier approach.

### 5.5.2 Specifications vs Input-Output Pairs

The second important ramification in formulating a program learning task is based on how the problem is conveyed to the network. Two directions are currently being extensively researched, one is to have specifications for solving the problem in natural language, the other is based on feeding the model with many input-output pairs.

There are also hybrid methods, where both types of information are presented to the learning model. While Yin and Neubig [10] present a method for inferring code from specifications, Balog et al. [7] and Parisotto et al., [23] perform program synthesis based on input-output pairs. The methods in Ling et al. [25] and Ling et al. [26] are examples of hybrid approaches.

### 5.5.3 End-to-End Learning vs Intermediate Steps Prediction

Yet a third difference in approaches to model program induction can be noticed in specialized literature on this topic: learning to predict the end result versus learning to generate a rationale for solving the task at hand.

For instance, Ling et al. [26] present a method for solving simple math problems described in natural language with multiple-choice answers. Besides predicting the correct result, the model also learns to generate a derivation of the answer through a series of small steps.

Both program synthesis and intermediate steps prediction can be modeled as sequence-to-sequence learning problems. They also describe a process to derive the end result. However, they seem to be conceptually different. Although code synthesis finds the process to arrive at a certain result, it does not give us any hint on how it arrived to that solution or program.

On the other hand, intermediate steps prediction forces the model to derive logical steps similar to the ones humans use in problem solving. This can have a great impact in understanding the choices that artificial learning models make.