## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 1. Abstract

Deep learning has seen great success in the automated extraction of facts from large and diverse sensory inputs. More recently there has been an increased interest in the development of models that also have the ability to reason at some level about these facts.

One such area of research is visual relational reasoning, which doesn't only aim to detect objects and their classes in a visual scene, but also aims to answer questions about these objects. Examples of this kind of modelling include the works of (Santoro et al. [5]) and (Hudson & Manning [4]). Their setups consist of 3D scenes of objects (CLEVR dataset, also see [3]), which are presented to a machine learning model that needs to answer questions involving relationships between the objects. Such a question could be: "What is the shape of the object behind the blue sphere?".

These kind of tasks pose challenges to traditional deep learning models because they need to abstract what consitutes an object and then they have to perform reasoning based on the object's position, features and relationships with the other objects. Therefore, all sorts of novel techniques (eg. paired processing, composition, attention, memory) need to be incorporated if such systems are to perform more complex reasoning. These are mainly used for exploiting relationships between objects as prior knowledge and thus, one of the goals of this research is to explore the possible ways in which relational reasoning can be integrated into deep learning models.

If we come to think of it, it would seem that relational reasoning in neural models touches some key points from the old connectionism vs symbolic manipulation AI debate, which is briefly explained in [8]. On one hand we need the adaptive power and fault tolerance properties of connectionist models to be able to process a wide range of sensory inputs, while on the other hand we require symbolic and computational abilities to make sense of the extracted facts and the relations between them in order to draw conclusions, which is what reasoning would seem to do at a very simplistic level.

Another area of research studies the broader aspect of integrating abstraction and reasoning into machine learning models. The recent work of (Chollet [2]) discusses aspects to be taken into account in order to define a measure for machine intelligence and proposes ARC, a dataset devised for testing whether a machine learning model incorporates various reasoning abilities.

The ARC dataset has recently been used for an online competition [1], where one can still test various approaches to the problem. The general idea is to train a model on numerous different tasks, each with a few samples of input-output pairs. For assessing the model's general reasoning abilities, a test dataset containing novel tasks is used. In this case, input-output pairs are represented as grids of different sizes, with cells taking a limited number of discrete values which are visualized as different colors. Although the setup defined appears minimalist, the tasks that can be defined on it are quite diverse. These include the fill algorithm, filling missing patterns, fractals, switching colors, etc.

More generally, the task of finding a series of instructions that map an input to an output given several training input-output pairs is referred to as program induction, or program synthesis if the model not only learns the mapping, but also outputs the program which performs the mapping. Otherwise, this is referred to as latent program induction - the model learns one or more programs, but the programs are not accessible in the form of series of instructions.

One of the earlier works in this field, (Balog et al. [7]) exploits deep learning to estimate the probability of a primitive existing inside a program. The primitives are defined within a DSL containing both first-order functions and high-order functions. The estimated probabilities are then used to guide a search for programs consistent with the provided input-output pairs.

More complex models of program synthesis have emerged recently, such as the work of (Ellis et al. [6]), which aims at creating systems that solve problems by writing programs. These programs map the initial input state to the desired final output state and overall the system can solve tasks from many domains, such as lists manipulation, formulas from physics and graphical patterns generation. Broadly speaking, the model is based on search, refactoring and the construction of libraries from learned concepts that are derived from primitives. These concepts are stored as modules and are reused for solving novel problems. The model proposed also has an imagination component, which is used for sampling and replaying problems.

Thus, we arrive at another goal of this research, which is to explore the role of relational reasoning in programming tasks. My view is that general program synthesis would require an AI model capable of deriving the relations between the objects it acts upon (eg. object oriented programming). Furthermore, based on cognitive studies, it would seem that a broad category of humans are visual problem solvers, so they could be deriving relations between key actors in the problem that they are solving by using visualization. This would suggest a strong link between visualization, relational reasoning and problem solving.

### 2. Plan

**Note**: This plan was designed for 2nd semester of the 2021 academic year and only the first 2 phases were carried out successfully. Resuming at phase 3.

|Start|Duration|Time|Phase|Description|
|:---:|:------:|:--:|:---:|:---------:|
|Mid Feb|2 weeks|10|Broad Research|Collect materials for thesis, discuss & decide theme|
|March|2 weeks|30|In-Depth Understanding & Practice|Study material in detail, check techniques involved, practice on sample code, comparison of past results|
|Mid March|6 weeks|28 + 14|Experiments & Evaluation of Results|Visual Rel., Abstr. & Reason., Deep Coder, Comparison on CLEVR, Neural Problem Solving|
|May|6 weeks|0|Thesis Elaboration|Designated for writing about findings, results, comparisons etc.|

#### Thesis Structure and Layout

**Note**: Need to get the required template

##### Abstract

- Todo at the end

##### 1. Outline

- Brief chapters description

##### 2. Introduction

*High level summary*

- Content from the executive abstract
- General background

##### 3. Related work

- Content from the in depth analysis of the selected bibliography
- Specific background and concepts

##### 4. Experimental results

*Functional description*

- Content from the experimental ideas
- Problem formulation, models, analysis of results and comparison with other studies

##### 5. Technical details

- Content from the practice report
- Models, parameters, limitations, scalability
- Libraries, modules

##### 6. Applications

- Tbd if possible

##### 7. Conclusions

- Todo at the end

##### Bibliography

### 3. Datasets

#### Base

1. ARC - input & output grids ([1], [2])
2. Synthetic/collected datasets - programs, inputs & outputs, primitives, DSLs ([6], [7])
3. CLEVR - images ([4], [5])
4. Open Images ([3])

#### Extended

|Dataset Name |Reference |Domain      |Input                   |Output                   |Description / Task (details) |Nat/Sync |
|:-----------:|:--------:|:----------:|:----------------------:|:-----------------------:|:---------------------------:|:-------:|
|Open Images  |[3]       |img         |img                     |bbox (pos) + rel (class) |rel between obj in img       |Yes      |
|CLEVR        |[4],[5]   |img, lang   |img + txt (question)    |word (class)             |rel vs non-rel questions     |Yes      |
|Sort of CLEVR|-         |img         |img + multi-class       |word (class)             |...                          |No       |
|bAbi         |[5]       |QA          |?                       |?                        |question - answer            |Yes.     |
|ARC          |[1],[2]   |?           |grid (multi-class)      |grid (multi-class)       |...                          |No       |
|Dream Coder  |[6]       |DSL & other |?                       |?                        |program generation           |No       |
|Deep Coder   |[7]       |DSL         |input-output pairs      |program from DSL         |program search               |No       |
|Alpha Code   |[15]      |source code |problem statement (txt) |code                     |program induction seq2seq    |Yes      |
|Eng - Fr tbd |[14],[16] |lang        |txt                     |txt                      |machine translation seq2seq  |Yes      |
|Img descript?|[18]      |img, lang   |img                     |txt                      |image description            |Yes      |

### 4. Report: Studies on the initial Bibliography

This report continues the study started in the [thesis proposal](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/1_exec_abstract.md) by developing an in-depth analysis of the selected bibliography. The review on related literature is carried out with regards to our previously set research goals.

**Overall Goals:**

- Research possible ways to integrate relational reasoning in deep learning models
- Investigate the role of relational reasoning in solving programming tasks

**Content**

[1. A Simple Neural Network Model for Relational Reasoning](https://arxiv.org/pdf/1706.01427.pdf)

[2. Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf)

[3. Recurrent Relational Networks](https://arxiv.org/pdf/1711.08028v4.pdf)

[4. Deep Coder: Learning to Write Programs](https://arxiv.org/pdf/1611.01989.pdf)

[5. Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/pdf/2006.08381.pdf)

[6. On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547.pdf)

#### 4.1 A simple Neural Network Model for Relational Reasoning

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

|Sort-of-CLEVR|CLEVR|
|:-----:|:---------:|
|![Sort-of-CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/sort_of_clevr.png)|![CLEVR](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/clevr.png)|
|What is the color of the nearest object to the yellow object?| What size is the cylinder that is left of the brown metal thing that is left of the big sphere?|
|Green|Small|

The second step is to examine how the make the object extraction question dependent. Questions are processed word-by-word (list of integers assigned to words) by an LSTM and the final state (an embedding of the question) is passed on to the RN together with the objects pair.

These triples are batched through an MLP for learning the relational function g (the output of which is a measure of relevance for object pairs). These outputs are then aggregated by a second MLP which learns the function f used for providing the answer to the question.

Finally, the information about what constitues objects and how to parse the question is backpropagated in the CNN and in the LSTM. This approach shows a 27% improvement in accuracy on the CLEVR  dataset compared to the state-of-the-art at the time of the publication and a 31% improvement in accuracy on the Sort-of-CLEVR dataset compared to a baseline CNN-MLP architecture.

Sample open source [code](https://github.com/clvrai/Relation-Network-Tensorflow) applying RN to the Sort-of-CLEVR dataset.

Code running logs: see [practice report](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/4_practice_report.md)

#### 4.2 Compositional Attention Networks for Machine Reasoning

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

Code running logs: see [practice report](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/4_practice_report.md)

#### 4.3 Recurrent Relational Networks

#### 4.4 Deep Coder: Learning to Write Programs

##### High-Level Summary

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

##### Programs Representation

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

#### 4.5 Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning

- a system that learns to solve problems by writing programs
- start by using primitives to learn programs based on input-output pairs
- adds up new symbolic abstractions (refactoring) based on imagined and replayed problems
- learns libraries of concepts
- builds expertise by creating programming languages for expressing domain concepts

Tasks:

- classic inductive programming tasks
- drawing pictures, building scenes
- rediscovers basics of modern functional programming, vector algebra, classical physics

#### 4.6 On the Measure of Intelligence

- the need to define & evaluate intelligence (capacity for reasoning) in a way that enables comparisons between 2 systems
- 2 broad historical conceptions:
	- AI: comprison of skills exhibited by AIs & humans at specific tasks
	- Psychology: levaraging modern insight into developmental cognitive psychology
- since skills are highly modulated by prior knowledge & experience, then unlimited priors & training data allow experiments to buy aribitrary levels of skills for a system
- proposed definition of intelligence: skill-acquisition efficiency
- highlight on concepts (critical pieces in intelligent systems) such as: scope, generalization difficulty, priors & experience
- overall, the study offers a more phylosophical, but grounded in research view on intelligence and makes a case for the design of general AI benchmark datasets  

### 5. Definitions and Notes

1. **Relational Reasoning**: the capacity to reason about and find solutions to problems involving a set of objects which are related to one another through some properties that need to be discovered.

2. **Program Induction**: the inference of computer programs designed to solve problems given in the form of partial data, either input-output pairs, or other types of constraints.

3. **Inductive Program Synthesis**: IPS problem, given input-output examples, produce a program that has behavior consistent with the examples. This requires solving 2 problems: defining the program space & the search procedure and solving the ranking problem - deciding which program is to be preferred when several solutions are available.

4. **Abstract Syntax Tree**: AST, a tree representation of the abstract syntactic structure of source code written in a programming language.

5. **Kernel**: Convolution matrix or mask, used for applying a filter to an image.

6. **Feature Map**: a collection of kernel activations which are the result of applying filters from one convolutional network layer to the next layer. The filters / kernels are represented by learnable weights, while the feature map is the activation of a CNN at an intermediary layer.

7. **Embedding**: a relatively low-dimensional space into which high-dimensional vectors can be translated.

8. **Latent Representation**: a representation of data which is available in a neural network's hidden layers. These representations fill a latent space, which can be viewed as an embedding when the network acts as an encoder (when it compresses data). To note the type of embeddings which retain semantic properties, such as Word2Vec (Mikolov et al. [9]).

9. **Beam Search**: a version of BFS, which uses a heuristic to only keep a subset of (best) partial solutions explored at any given point during the search process.

10. **BLEU Score**: bilingual evaluation understudy is a metric for evaluating machine translated text.

### 6. A Comparison of relational and compositional machine learning models (part 1)

Based on the literature studied so far (and the experiments performed in 2019) we can start highlighting some similarities between the deep learning models which aim to perform relational reasoning / inference. Generally, the shortcomings in capturing relational properties from a dataset by a machine learning model are due to the lack of proper design and/or prior knowledge. However, this problem manifests itself differently depending on the case. Let us take a look at 2 well known examples.

#### 6.1 Convolution Maps

In visual recognition tasks, CNNs outperform MLPs simply because they exploit spatial relationships. Instead of having fully connected layers with different learnable weights, CNNs have shared weights (kernels / convolutions) and thus learn locally invariant features, meaning that the same properties (edges, textures, etc.) are learnt across every region of the image, whereas a MLP would not have this constraint and thus would have the potential to overfit specific properties of a given region. For this reason, we can conclude that CNNs have the proper design to learn (fit) relations between pixels and regions in images which are generally (and not only locally) useful for computing the required output in a visual task.

#### 6.2 Attention Functions

The next example is a machine translation model, namely the sequence to sequence modelling, where a recurrent neural network is fed an input sequence and has to produce an output sequence, such as translating a sentence from english to french (Sutskever et al. [14]) or synthesizing a program from a problem description (Li et al. [15]). One major breakthrough in this area was the use of an attention function (Bahdanau et al. [16]). Various implementations of attention models (Shazeer [17]) and visual attention (Xu et al. [18]). The seq2seq model was initially designed as an encoder-decoder architecture, where an RNN would process the input and provide a vector / state for the decoder to decode into the output.

One problem with this model was that it was not capable to properly encode longer sentences into a finite hidden state (fixed length vector) at the end of processing the input. And so essential information would be lost this way. The attention function alters this behaviour by constraining the decoder to attend to the hidden states of the encoder in a finite subsequence around the target word (interval), thus providing a context at each step in the output sequence by utilizing potential relations between consecutive words. The practical consequence of this modification is an enhanced ability to correctly learn to generate larger sequences (improved generalization capabilities).

|Seq2seq|Info|
|:-:|:---------:|
|![Encoder-Decoder versions](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/encoder_decoder.png)|**Encoder-Decoder versions**. (a) Vanilla Encoder-Decoder: only the final hidden state of the encoder is passed on to the decoder as initial input. (b) Attention based Encoder-Decoder: intermediary hidden states from the encoder are weighted in according to an attention function and fed into the decoder at all steps.|
|![Attention function](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attention_function.png)|**Attention function**: a more detailed view of the mechanism and computation.|

|Attention matrix|Info|
|:-:|:---------:|
|![Attention matrix](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attention_matrix.png)|Source: Bahdanau et al. [16]. Displays how much should the hidden state obtained when processing the j-th english word contribute to predicting the i-th french word.|

|MAC Attention map|Info|
|:-:|:---------:|
|![MAC Attention map](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/mac_attention.png)|Source: Hudson & Manning [4]. 3-steps reasoning (MAC network of length 3) based on the interaction of the memory, attention and control units.|

#### 6.3 Tutorials and supporting documentation

[A. LSTM, Encoder-Decoder and Attention](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)[+](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)

[B. Andrew Ng Tutorials](https://www.youtube.com/watch?v=RLWuzLLSIgw)

### 7. A Comparison of relational and compositional machine learning models (part 2)

#### 7.1 SortNet vs. RN (Santoro et al. [5])

**SortNet steps** (2019 experiments)

- Design: eg. using an adequate neural network structure
- Prior Knowledge: eg. using the order relations as input
- Relational Network: Design + Prior Knowledge - integrating the prior knowledge into a neural network's design without data transformation 

**Problem Formulation**

TODO

**Comparison**

Both the SortNet and the RN learn relations between objects in the input, but the learning is modelled in a slightly different way.

- Using convolutions to represent relations in the SortNet case.
- Using pairs as separate training data for a relational function in the RN case.

Debate on how MLP would implement logic in RN.

#### 7.2 RN (Santoro et al. [5]) vs MAC (Hudson & Manning [4])

The relational function learned at the level of paired objects (CNN feature maps) in the RN model is very similar at the conceptual level with the attention function learned in the seq2seq model for pairs of words in different languages.

### 8. Directions in Program Induction

#### 8.1 Latent Induction vs Code Synthesis

The first important bifurcation in the approaches for generating programs is the choice of program representation. When a neural network learns to map input to output, thus solving a programming task, the program is stored in the network and executed by the network through neural activations. This is called latent program induction, because the representation of the generated program is not human-readable.

The second choice is to formulate the problem such that the neural network outputs a program in a language, which is then executed to get the desired output. This is generally referred to as program synthesis.

A comparison of the two approaches applied on string transformation problems is carried in Devlin et al. [21].

Latent programs are written in the language of neural networks, whereas synthesized programs are written in a language of choice. Both approaches have shown success, however it is not possible to pick one that works best because they have different strengths. For instance, induction is more likely to provide a good approximation of the output function for the type of inputs provided, but might not generalize so well for new inputs. On the other hand, synthesis will either find the correct program and generalize the solution well for all inputs, or find the wrong solution which over-fits the presented input. Synthesis is thus more capable, but also the riskier approach.

#### 8.2 Specifications vs Input-Output Pairs

The second important ramification in formulating a program learning task is based on how the problem is conveyed to the network. Two directions are currently being extensively researched, one is to have specifications for solving the problem in natural language, the other is based on feeding the model with many input-output pairs.

There are also hybrid methods, where both types of information are presented to the learning model. While Yin and Neubig [10] present a method for inferring code from specifications, Balog et al. [7] and Parisotto et al., [23] perform program synthesis based on input-output pairs. The methods in Ling et al. [25] and Ling et al. [26] are examples of hybrid approaches.

#### 8.3 End-to-End Learning vs Intermediate Steps Prediction

Yet a third difference in approaches to model program induction can be noticed in specialized literature on this topic: learning to predict the end result versus learning to generate a rationale for solving the task at hand.

For instance, Ling et al. [26] present a method for solving simple math problems described in natural language with multiple-choice answers. Besides predicting the correct result, the model also learns to generate a derivation of the answer through a series of small steps.

Both program synthesis and intermediate steps prediction can be modeled as sequence-to-sequence learning problems. They also describe a process to derive the end result. However, they seem to be conceptually different. Although code synthesis finds the process to arrive at a certain result, it does not give us any hint on how it arrived to that solution or program.

On the other hand, intermediate steps prediction forces the model to derive logical steps similar to the ones humans use in problem solving. This can have a great impact in understanding the choices that artificial learning models make.

### 9. Relational reasoning and question answering in programming

#### 9.1 Relational input-output pairs

Consider input-output pairs in IPS to be program states, which we can generate embeddings for. If the program attributes that need to be estimated were to be relational, then an RN could in theory improve the MLP used for estimating the program attributes. This item would be worth testing in a setup where relational program attributes could somehow be used to optimize the program search.

#### 9.2 Program attributes as questions

The cognitive process of designing a program to solve a problem is a highly complex task. Often times, it is a longer interactive process during which the solver has to ask a series of questions in order to arrive at the right programming technique and abstractions through meaningful decisions. Thus, the ability to ask meaningful questions seems to be a necessary component when trying to design a more general reasoning system. To research how questions could be generated in a programming setup?

#### 9.3 Abstract Syntax Tree base Neural Networks

A. [A Novel Neural Source Code Representation based on Abstract Syntax Tree, 2019](http://xuwang.tech/paper/astnn_icse2019.pdf), Zhang et al. [19]

- treating code as natural language texts fails to capture syntactical and semantic information
- more long term dependencies are required in modelling source code
- tasks: code clone detection and source code classification

B. [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree, 2020](https://arxiv.org/pdf/2002.08653.pdf), Wang et al. [20]

- difference between syntactic and semantic clones
- flow augmented abstract syntax trees, data flow
- datasets: google code jam and big clone bench
- graph neural networks

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
def neural_net(x, num_classes, num_labels, dropout, reuse, is_training):
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

### 11. References

1. [Abstraction & Reasoning Challenge, Kaggle 2020](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)
2. [On the Measure of Intelligence, 2019](https://arxiv.org/pdf/1911.01547.pdf)
3. [Open Images - Visual Relationship, Kaggle 2019](https://www.kaggle.com/c/open-images-2019-visual-relationship/)
4. [Compositional Attention Networks for Machine Reasoning, 2018](https://arxiv.org/pdf/1803.03067.pdf)
5. [A Simple Neural Network Model for Relational Reasoning, 2017](https://arxiv.org/pdf/1706.01427.pdf)
6. [Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning, 2020](https://arxiv.org/pdf/2006.08381.pdf)
7. [Deep Coder: Learning to Write Programs, 2017](https://arxiv.org/pdf/1611.01989.pdf)
8. [Connectionism vs. computationalism debate, Wikipedia](https://en.wikipedia.org/wiki/Connectionism)
9. [Distributed Representations of Words and Phrases and their Compositionality, 2013](https://arxiv.org/pdf/1310.4546.pdf)
10. [A Syntactic Neural Model for General-Purpose Code Generation, 2017](https://arxiv.org/pdf/1704.01696.pdf)
11. [The Sketching Approach to Program Synthesis, 2008](https://people.csail.mit.edu/asolar/papers/Solar-Lezama09.pdf)
12. [Hybrid computing using a neural network with dynamic external memory, 2016](https://www.nature.com/articles/nature20101) [+](https://deepmind.com/blog/article/differentiable-neural-computers)
13. [Combinatorial Optimization and Reasoning with Graph Neural Networks, 2021](https://arxiv.org/pdf/2102.09544.pdf)
14. [Sequence to Sequence Learning with Neural Networks, 2014](https://arxiv.org/pdf/1409.3215.pdf)
15. [Competition-Level Code Generation with AlphaCode, 2022](https://storage.googleapis.com/deepmind-media/AlphaCode/competition_level_code_generation_with_alphacode.pdf) [+](https://www.deepmind.com/blog/competitive-programming-with-alphacode)
16. [Neural Machine Translation by Jointly Learning to Align and Translate, 2015](https://arxiv.org/pdf/1409.0473.pdf)
17. [Fast Transformer Decoding, 2019](https://arxiv.org/pdf/1911.02150.pdf)
18. [Show, Attend and Tell, 2016](https://arxiv.org/pdf/1502.03044.pdf)
19. [A Novel Neural Source Code Representation based on Abstract Syntax Tree, 2019](http://xuwang.tech/paper/astnn_icse2019.pdf)
20. [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree, 2020](https://arxiv.org/pdf/2002.08653.pdf)
21. [RobustFill: Neural Program Learning under Noisy I/O, 2017](https://arxiv.org/pdf/1703.07469.pdf)
22. [Neural Program Meta-Induction, 2017, (Devlin et al.)](https://arxiv.org/pdf/1710.04157.pdf)
23. [Neuro-Symbolic Program Synthesis, 2016](https://arxiv.org/pdf/1611.01855.pdf)
24. [Recurrent Relational Networks, 2018, (Palm et al.)](https://paperswithcode.com/paper/recurrent-relational-networks)[+](https://arxiv.org/pdf/1711.08028v4.pdf)
25. [Latent Predictor Networks for Code Generation, 2016](https://aclanthology.org/P16-1057.pdf)
26. [Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems, 2017](https://aclanthology.org/P17-1015.pdf)