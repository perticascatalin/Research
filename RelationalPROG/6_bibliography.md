# Relational reasoning in deep learning: a parallel between solving visual and programming tasks

## 11. References

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
24. [Recurrent Relational Networks, 2018](https://paperswithcode.com/paper/recurrent-relational-networks)[+](https://arxiv.org/pdf/1711.08028v4.pdf)
25. [Latent Predictor Networks for Code Generation, 2016](https://aclanthology.org/P16-1057.pdf)
26. [Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems, 2017](https://aclanthology.org/P17-1015.pdf)

### Definitions and Notes

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