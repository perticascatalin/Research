## Future Work

### In-progress Work

*To investigate further & turn these into tickets*

1. **Representation**

Sequence of numbers vs. bag of numbers (count sort)

2. **Multilabel Classification**

Vectors for small arrays of numbers. 

Largest enhancements and smallest shrinking preserve array class (all labels).

Bent surface on n-2 space.

3. **Regression**

Directly predict the numbers in the correct order.

Neural network has to store the numbers.

4. **Temporal Generation. Sequence to sequence. Results which can be compared to previous papers on read-write NN.**

5. **Pre-order to/vs. Post-order**

6. **Constraint non-identical labels**

7. **Trick network by having some fixed positions.**

K out of N elements are fixed such that solving the problem for them yields better results than for bothering with the rest of the elements.

### Experiments vs Theory

1. Adding comparison prior knowledge to data - DONE

2. Models scalability - DONE

3. Problem Complexity - DONE

4. Storage capability: predicting the order vs. predicting elements in order

5. Maximizing expected value vs. optimization

6. Impurity vs. local minima

7. Attribute split vs. non-linear combinations


### Visualization of results

- **barchart-done**
- stacking
- coloring

Display graphic array before and after: 

### Analysis of results

Where does the model fail?

Split into categories [useless (less than 10% acc) - guess (max 5) - good guess (7) - problem solved (N)].

Rank all the samples in a batch - by accuracy or mean squared error (proximity).

### Harder Stuff

Looking at the input in a sequential manner or deriving rules for correct input parsing - sorted numbers should be easily mapped to 0, 1, 2, 3, ... what if many numbers from a random permutation of numbers? Distributive attention - how to paralelly process 2 different data streams.

Learning specific permutations and the mapping within them through a Cayley Graph. Distributing the types of inputs to different models goes in the direction of AutoML.

### Num samples debate

Depending on num_classes, the num_samples should be above a certain threshold for decent accuracy

The threshold can be computed using N! = num_classes!

- N!/num_samples < T
- 6! = 720
- 1000 decent sample (on par)
- 8! = 40320
- 10000 decent sample (generalize full for 4 classes)
- 10! = 3628800
- 60000 decent sample (generalize well for 60 classes)
- 12! = 479001600

what is a decent sample size?
20!/100.000 = 2.4e+13

### Principles of writing

- keep theory & speculation in balance with maths and experimental part
- link neighboring points through story and insight
- keep flat structure, so that it is clear where to add certain stuff
- polish partial parts, but leave way for new parts to be created