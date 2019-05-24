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