# Relational reasoning in deep learning: a parallel between solving visual and programming tasks

## 8. Comparison of relational and compositional machine learning models

### 8.1 SortNet vs. RN (Santoro et al. [5])

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

### 8.2 RN (Santoro et al. [5]) vs MAC (Hudson & Manning [4])

The relational function learned at the level of paired objects (CNN feature maps) in the RN model is very similar at the conceptual level with the attention function learned in the seq2seq model for pairs of words in different languages.

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