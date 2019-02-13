# MastersExperiments

## Experiment 1: Statistically learning the correct order

### Models

Comparison between decision tree and multilayer perceptron

Data + Comparisons

Multiple variants of decision trees: decision, forest and extreme

Some results

- Data: DC
- Range 8: D 6.0, E_96 7.9, NN 8.0
- Range 10: D 4.0 E_96 7.5 NN 9.5
- Range 16: NN 3.6 E_96 5.4
- Range 20: NN 1.4 E_96 4.6

Maximizing expected value vs. optimization

Attribute split vs. non-linear combinations

### Related work

Links to information theory

There are some other works where simple algorithms are inferred via neural networks. For example, in [10], the operations of copying and sorting an array are performed with a fully differentiable network connected to an eternal memory via attention.

[10] A. Graves, G. Wayne, and I. Danihelka, “Neural Turing Machines,” arXiv:1410.5401v2, 2014.

In another approach, [11] presents a sorting experiment for sets of numbers using a sequence to sequence model.

[11] O. Vinyals, S. Bengio, and M. Kudlur, “OrderMatters: Sequence to Sequence for Sets”, in 4th International Conference
on Learning Representations (ICLR), 2016.

The importance of prior information and pre-learned intermediate concepts. Composition of 2 highly non-linear tasks. [x]

[x] C. Gulcehre and Y. Bengio, "Knowledge Matters: Importance of Prior Information for Optimization"