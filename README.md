# MastersExperiments

## Experiment 1: Statistically learning the correct order (in progress)

The experiment consists of taking arrays with different numbers and of different lenghts and formulating the problem of predicting the sorted order of the initial numbers. We show how this poses scalability problems for various machine learning models and try to find some of the underlying reasons. Additionaly we look for changes in the models or the problem formulation that could help improve our solution.

### Models

Comparison between decision tree and multilayer perceptron

Data + Comparisons

Multiple variants of decision trees: decision, forest and extreme

Representation: sequence of numbers vs. bag of numbers (count sort)

Regression

Temporal Generation

Pre-order vs. Post-order

**Some results**

Metrics: We compute a partial accuracy - the average number of elements guessed in N arrays.

- Data: DC
- Range 8: D 6.0, E_96 7.9, NN 8.0
- Range 10: D 4.0 E_96 7.5 NN 9.5
- Range 16: NN 3.6 E_96 5.4
- Range 20: NN 1.4 E_96 4.6

Views: display graphic array before and after - stacking vs. coloring.

### Experiments vs Theory

Adding comparison prior knowledge to data

Storage capability: predicting the order vs. predicting elements in order

Maximizing expected value vs. optimization

Impurity vs. local minima

Attribute split vs. non-linear combinations

Models scalability

### Related work

Links to information theory and deep learning [X], [Y].

The importance of prior information and pre-learned intermediate concepts. Composition of 2 highly non-linear tasks and other hypothesis such as local minima obstacle and guided/transfer learning [X].

Gradients in highly composed functions or hard constraints [Y].

There are some other works where simple algorithms are inferred via neural networks. For example, in [A], the operations of copying and sorting an array are performed with a fully differentiable network connected to an eternal memory via attention. 

In another approach, [B] presents a sorting experiment for sets of numbers using a sequence to sequence model.

[A] A. Graves, G. Wayne, and I. Danihelka, “Neural Turing Machines,” arXiv:1410.5401v2, 2014.

[B] O. Vinyals, S. Bengio, and M. Kudlur, “OrderMatters: Sequence to Sequence for Sets”, in 4th International Conference on Learning Representations (ICLR), 2016.

[X] C. Gulcehre and Y. Bengio, "Knowledge Matters: Importance of Prior Information for Optimization"

[Y] S. Shalev-Shwartz and O. Shamir and S Shammah, "Failures of Gradient-Based Deep Learning"