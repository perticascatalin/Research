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
|Mid March|6 weeks|16|Experiments & Evaluation of Results|Visual Rel., Abstr. & Reason., Deep Coder, Comparison on CLEVR, Neural Problem Solving|
|May|6 weeks|0|Thesis Elaboration|Designated for writing about findings, results, comparisons etc.|

### 3. Datasets

1. ARC - input & output grids (1, 2)
2. Synthetic/collected datasets - programs, inputs & outputs, primitives, DSLs (6, 7)
3. CLEVR - images (4, 5)
4. Open Images (3)

### 4. References

1. [Abstraction & Reasoning Challenge, Kaggle 2020](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)
2. [On the Measure of Intelligence, 2019](https://arxiv.org/pdf/1911.01547.pdf)
3. [Open Images - Visual Relationship, Kaggle 2019](https://www.kaggle.com/c/open-images-2019-visual-relationship/)
4. [Compositional Attention Networks for Machine Reasoning, 2018](https://arxiv.org/pdf/1803.03067.pdf)
5. [A simple Neural Network Model for Relational Reasoning, 2017](https://arxiv.org/pdf/1706.01427.pdf)
6. [Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning, 2020](https://arxiv.org/pdf/2006.08381.pdf)
7. [Deep Coder: Learning to Write Programs, 2017](https://arxiv.org/pdf/1611.01989.pdf)
8. [Connectionism vs. computationalism debate, Wikipedia](https://en.wikipedia.org/wiki/Connectionism)