## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### Abstract

Deep learning has seen great success in the automated extraction of facts from large and diverse sensory inputs. More recently there has been an increased interest in the development of models that also have the ability to reason at some level about these facts.

One such area of research is visual relational reasoning, which doesn't only aim to detect objects and their classes in a visual scene, but to answer questions about these objects as well. Examples of this kind of modelling include the works of (Santoro et al. [5]) and (Hudson & Manning [4]). Their setups consist of 3D scenes of objects which are presented to a machine learning model that needs to answer questions involving relationships between the objects. Such a question could be: "What is the shape of the object behind the blue sphere?".

These kind of tasks pose challenges to traditional deep learning models because they need to abstract what consitutes an object and then they have to perform reasoning based on the object's position, features and relationships with the other objects. Therefore, all sorts of novel techniques (eg. paired processing, composition, attention, memory) need to be incorporated if such systems are to perform more complex reasoning. These are mainly used for exploiting relationships between objects as prior knowledge and thus, one of the goals of this research is to explore the possible ways in which relational reasoning can be integrated into deep learning models.

If we come to think of it, it would seem that relational reasoning in neural models touches some key points from the old connectionism vs symbolic manipulation debate, which is briefly explained in [8]. On one hand we need the adaptive power and fault tolerance properties of connectionist models to be able to process a wide range of sensory inputs, while on the other hand we require symbolic and computational abilities to make sense of the extracted facts and the relations between them to draw conclusions, which is what reasoning would seem to do at a very simplistic level.

### Ideas

- abstraction and reasoning in a minimalist setup, to model a measure for intelligence of AI systems
- solving a task by writing a program which maps input (initial state) to output (finish state) from dream coder
- using deep learning to estimate the probability of a primitive existing inside a program from deep coder
- for general program synthesis, it is necessary that an AI can derive the relations between the objects it acts on (eg. object oriented programming)
- neural problem solving, operations inside neural networks

### Datasets

1. ARC - input & output grids (1, 2)
2. Synthetic/collected datasets - programs, inputs & outputs, primitives, DSLs (6, 7)
3. CLEVR - images (4, 5)
4. Open Images (3)

### References

1. [Abstract Reasoning Challenge, Kaggle 2020](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)
2. [On the Measure of Intelligence, 2019](https://arxiv.org/pdf/1911.01547.pdf)
3. [Open Images - Visual Relationship, Kaggle 2019](https://www.kaggle.com/c/open-images-2019-visual-relationship/)
4. [Compositional Attention Networks for Machine Reasoning, 2018](https://arxiv.org/pdf/1803.03067.pdf)
5. [A simple Neural Network Model for Relational Reasoning, 2017](https://arxiv.org/pdf/1706.01427.pdf)
6. [Dream Coder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning, 2020](https://arxiv.org/pdf/2006.08381.pdf)
7. [Deep Coder: Learning to Write Programs, 2017](https://arxiv.org/pdf/1611.01989.pdf)
8. [Connectionism vs. computationalism debate, Wikipedia](https://en.wikipedia.org/wiki/Connectionism)