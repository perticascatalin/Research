# Relational reasoning in deep learning: a parallel between solving visual and programming tasks

## 7. A Comparison of relational and compositional machine learning models

### 7.1 SortNet vs. RN (Santoro et al. [5])

**SortNet steps** (2019 experiments)

- Design: eg. using an adequate neural network structure
- Prior Knowledge: eg. using the order relations as input
- Relational Network: Design + Prior Knowledge - integrating the prior knowledge into a neural network's design without data transformation 

**Problem Formulation**

TODO, see Practice Report

**Comparison**

Both the SortNet and the RN learn relations between objects in the input, but the learning is modelled in a slightly different way.

- Using convolutions to represent relations in the SortNet case.
- Using pairs as separate training data for a relational function in the RN case.

Debate on how MLP would implement logic in RN.

### 7.2 RN (Santoro et al. [5]) vs MAC (Hudson & Manning [4])

The relational function learned at the level of paired objects (CNN feature maps) in the RN model is very similar at the conceptual level with the attention function learned in the seq2seq model for pairs of words in different languages.

## 8. Directions in Program Induction

### 8.1 Latent Induction vs Code Synthesis

The first important bifurcation in the approaches for generating programs is the choice of program representation. When a neural network learns to map input to output, thus solving a programming task, the program is stored in the network and executed by the network through neural activations. This is called latent program induction, because the representation of the generated program is not human-readable.

The second choice is to formulate the problem such that the neural network outputs a program in a language, which is then executed to get the desired output. This is generally referred to as program synthesis.

A comparison of the two approaches applied on string transformation problems is carried in Devlin et al. [21].

Latent programs are written in the language of neural networks, whereas synthesized programs are written in a language of choice. Both approaches have shown success, however it is not possible to pick one that works best because they have different strengths. For instance, induction is more likely to provide a good approximation of the output function for the type of inputs provided, but might not generalize so well for new inputs. On the other hand, synthesis will either find the correct program and generalize the solution well for all inputs, or find the wrong solution which over-fits the presented input. Synthesis is thus more capable, but also the riskier approach.

### 8.2 Specifications vs Input-Output Pairs

The second important ramification in formulating a program learning task is based on how the problem is conveyed to the network. Two directions are currently being extensively researched, one is to have specifications for solving the problem in natural language, the other is based on feeding the model with many input-output pairs.

There are also hybrid methods, where both types of information are presented to the learning model. While Yin and Neubig [10] present a method for inferring code from specifications, Balog et al. [7] and Parisotto et al., [23] perform program synthesis based on input-output pairs. The methods in Ling et al. [25] and Ling et al. [26] are examples of hybrid approaches.

### 8.3 End-to-End Learning vs Intermediate Steps Prediction

Yet a third difference in approaches to model program induction can be noticed in specialized literature on this topic: learning to predict the end result versus learning to generate a rationale for solving the task at hand.

For instance, Ling et al. [26] present a method for solving simple math problems described in natural language with multiple-choice answers. Besides predicting the correct result, the model also learns to generate a derivation of the answer through a series of small steps.

Both program synthesis and intermediate steps prediction can be modeled as sequence-to-sequence learning problems. They also describe a process to derive the end result. However, they seem to be conceptually different. Although code synthesis finds the process to arrive at a certain result, it does not give us any hint on how it arrived to that solution or program.

On the other hand, intermediate steps prediction forces the model to derive logical steps similar to the ones humans use in problem solving. This can have a great impact in understanding the choices that artificial learning models make.

## 9. Relational reasoning and question answering in programming

### 9.1 Relational input-output pairs

Consider input-output pairs in IPS to be program states, which we can generate embeddings for. If the program attributes that need to be estimated were to be relational, then an RN could in theory improve the MLP used for estimating the program attributes. This item would be worth testing in a setup where relational program attributes could somehow be used to optimize the program search.

### 9.2 Program attributes as questions

The cognitive process of designing a program to solve a problem is a highly complex task. Often times, it is a longer interactive process during which the solver has to ask a series of questions in order to arrive at the right programming technique and abstractions through meaningful decisions. Thus, the ability to ask meaningful questions seems to be a necessary component when trying to design a more general reasoning system. To research how questions could be generated in a programming setup?

### 9.3 Abstract Syntax Tree base Neural Networks

A. [A Novel Neural Source Code Representation based on Abstract Syntax Tree, 2019](http://xuwang.tech/paper/astnn_icse2019.pdf), Zhang et al. [19]

- treating code as natural language texts fails to capture syntactical and semantic information
- more long term dependencies are required in modelling source code
- tasks: code clone detection and source code classification

B. [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree, 2020](https://arxiv.org/pdf/2002.08653.pdf), Wang et al. [20]

- difference between syntactic and semantic clones
- flow augmented abstract syntax trees, data flow
- datasets: google code jam and big clone bench
- graph neural networks