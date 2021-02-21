## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### Abstract

Deep learning has seen great success in the automated extraction of facts from large and diverse sensory inputs. More recently there has been an increased interest in the development of models that also have the ability to reason at some level about these facts.

One such area of research is visual relational reasoning, which aims not only to detect objects and their classes in a visual scene, but to answer questions about these objects as well. Examples of this kind of modelling include the works of (Santoro et al. [5]) and (Hudson & Manning [4]). Their setups consist of 3D scenes of objects which are presented to a machine learning model that needs to answer questions involving relationships between the objects. Such a question could be:  "What is the shape of the object behind the blue sphere?". These kind of tasks pose challenges to traditional deep learning models because they need to abstract what consitutes and object and then they have to perform reasoning based on the object's position, features and relationships with the other objects.

If we come to think of it, ... connectionism vs symbolic manipulation paragraph.

### Ideas

- abstraction and reasoning in a minimalist setup, to model a measure for intelligence of AI systems
- answering questions based on scenes with objects
- solving a task by writing a program which maps input (initial state) to output (finish state) from dream coder
- exploiting relations between objects as prior knowledge
- integrating relational reasoning into deep learning models
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