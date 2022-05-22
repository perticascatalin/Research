# Relational reasoning in deep learning: a parallel between solving visual and programming tasks

## 1. Abstract

Deep learning has seen great success in the automated extraction of facts from large and diverse sensory inputs. More recently there has been an increased interest in the development of models that also have the ability to reason at some level about these facts.

One such area of research is visual relational reasoning, which doesn't only aim to detect objects and their classes in a visual scene, but also aims to answer questions about these objects. Examples of this kind of modelling include the works of (Santoro et al. [5]) and (Hudson & Manning [4]). Their setups consist of 3D scenes of objects (CLEVR dataset, also see [3]), which are presented to a machine learning model that needs to answer questions involving relationships between the objects. Such a question could be: "What is the shape of the object behind the blue sphere?".

These kind of tasks pose challenges to traditional deep learning models because they need to abstract what constitutes an object and then they have to perform reasoning based on the object's position, features and relationships with the other objects. Therefore, all sorts of novel techniques (eg. paired processing, composition, attention, memory) need to be incorporated if such systems are to perform more complex reasoning. These are mainly used for exploiting relationships between objects as prior knowledge or to reduce information bottlenecks, and thus, one of the goals of this research is to explore the possible ways in which relational reasoning can be integrated into deep learning models.

If we come to think of it, it would seem that relational reasoning in neural models touches some key points from the old connectionism vs symbolic manipulation AI debate, which is briefly explained in [8]. On one hand we need the adaptive power and fault tolerance properties of connectionist models to be able to process a wide range of sensory inputs, while on the other hand we require symbolic and computational abilities to make sense of the extracted facts and the relations between them in order to draw conclusions, which is what reasoning would seem to do at a very simplistic level.

Another area of research studies the broader aspect of integrating abstraction and reasoning into machine learning models. The recent work of (Chollet [2]) discusses aspects to be taken into account in order to define a measure for machine intelligence and proposes ARC, a dataset devised for testing whether a machine learning model incorporates various reasoning abilities.

The ARC dataset has recently been used for an online competition [1], where one can still test various approaches to the problem. The general idea is to train a model on numerous different tasks, each with a few samples of input-output pairs. For assessing the model's general reasoning abilities, a test dataset containing novel tasks is used. In this case, input-output pairs are represented as grids of different sizes, with cells taking a limited number of discrete values which are visualized as different colors. Although the setup defined appears minimalist, the tasks that can be defined on it are quite diverse. These include the fill algorithm, filling missing patterns, fractals, switching colors, etc.

More generally, the task of finding a series of instructions that map an input to an output given several training input-output pairs is referred to as program induction, or program synthesis if the model not only learns the mapping, but also outputs the program which performs the mapping. Otherwise, this is referred to as latent program induction - the model learns one or more programs, but the programs are not accessible in the form of series of instructions.

One of the earlier works in this field, (Balog et al. [7]) exploits deep learning to estimate the probability of a primitive existing inside a program. The primitives are defined within a DSL containing both first-order functions and high-order functions. The estimated probabilities are then used to guide a search for programs consistent with the provided input-output pairs.

More complex models of program synthesis have emerged recently, such as the work of (Ellis et al. [6]), which aims at creating systems that solve problems by writing programs. These programs map the initial input state to the desired final output state and overall the system can solve tasks from many domains, such as lists manipulation, formulas from physics and graphical patterns generation. Broadly speaking, the model is based on search, refactoring and the construction of libraries from learned concepts that are derived from primitives. These concepts are stored as modules and are reused for solving novel problems. The model proposed also has an imagination component, which is used for sampling and replaying problems.

Thus, we arrive at another goal of this research, which is to explore the role of relational reasoning in programming tasks. My view is that general program synthesis would require an AI model capable of deriving the relations between the objects it acts upon (eg. object oriented programming). Furthermore, based on cognitive studies, it would seem that a broad category of humans are visual problem solvers, so they could be deriving relations between key actors in the problem that they are solving by using visualization. This would suggest a strong link between visualization, relational reasoning and problem solving.

## 2. Plan

**Note**: This plan was designed for 2nd semester of the 2021 academic year and only the first 2 phases were carried out successfully. Resuming at phase 3.

|Start|Duration|Time|Phase|Description|
|:---:|:------:|:--:|:---:|:---------:|
|Mid Feb|2 weeks|10|Broad Research|Collect materials for thesis, discuss & decide theme|
|March|2 weeks|30|In-Depth Understanding & Practice|Study material in detail, check techniques involved, practice on sample code, comparison of past results|
|Mid March|6 weeks|32 + 18 + 8|Experiments & Evaluation of Results|Visual Rel., Abstr. & Reason., Deep Coder, Comparison on CLEVR, Neural Problem Solving|
|May|6 weeks|10|Thesis Elaboration|Designated for writing about findings, results, comparisons etc.|

### 2.1 Thesis Structure and Layout

**Note**: Need to get the required template

#### Abstract

- Todo at the end

#### 1. Outline

- Brief chapters description

#### 2. Introduction

*High level summary*

- Content from the executive abstract (**Section 1**, see 2.2)
- General background

#### 3. Related work

- Content from the in depth analysis of the selected bibliography (**Sections 4 - 5**, see 2.2)
- Specific background and concepts

#### 4. Method

*Functional description*

- Content from problem formulation (**Section 6**, see 2.2)
- Problem formulation, theory, models

#### 5. Experimental results

- Content from the experimental ideas (**Sections 8 - 9**, see 2.2)
- Analysis of results and comparison with other studies

#### 6. Technical details

- Content from the practice report (**Section 10**, see 2.2)
- Models, parameters, limitations, scalability
- Libraries, modules

#### 7. Applications

- Tbd if possible

#### 8. Conclusions

- Todo at the end

#### Bibliography

### 2.2 Current Report Sections

#### Section  1: Abstract
#### Section  2: Plan
#### Section  3: Datasets
#### Section  4: Report: Studies on the initial Bibliography
#### Section  5: Directions in Program Induction
#### Section  6: Design of models and connections
#### Section  7: Definitions and Notes
#### Section  8: Comparison of relational and compositional machine learning models
#### Section  9: Relational reasoning and question answering in programming
#### Section 10: Report on practice (technical details)
#### Section 11: References

## 3. Datasets

### Base

1. ARC - input & output grids ([1], [2])
2. Synthetic/collected datasets - programs, inputs & outputs, primitives, DSLs ([6], [7])
3. CLEVR - images ([4], [5])
4. Open Images ([3])

### Extended

|Dataset Name |Reference |Domain      |Input                   |Output                   |Description / Task (details) |Nat/Sync |
|:-----------:|:--------:|:----------:|:----------------------:|:-----------------------:|:---------------------------:|:-------:|
|Open Images   |[3]       |img         |img                     |bbox (pos) + rel (class) |rel between obj in img       |Yes      |
|CLEVR         |[4],[5]   |img, lang   |img + txt (question)    |word (class)             |rel vs non-rel questions     |Yes      |
|Sort of CLEVR |-         |img         |img + multi-class       |word (class)             |...                          |No       |
|bAbi          |[5]       |QA          |?                       |?                        |question - answer            |Yes.     |
|ARC           |[1],[2]   |?           |grid (multi-class)      |grid (multi-class)       |...                          |No       |
|Dream Coder   |[6]       |DSL & other |?                       |?                        |program generation           |No       |
|Deep Coder    |[7]       |DSL         |input-output pairs      |program from DSL         |program search               |No       |
|Alpha Code    |[15]      |source code |problem statement (txt) |code                     |program induction seq2seq    |Yes      |
|WWT'14(Eng-Fr)|[14],[16] |lang        |txt                     |txt                      |machine translation seq2seq  |Yes      |
|Flickr        |[18]      |img, lang   |img                     |txt                      |image caption            |Yes      |
|MS-COCO       |[18]      |img, lang   |img                     |txt                      |image caption            |Yes      |