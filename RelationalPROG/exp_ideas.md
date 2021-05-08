## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 10. Experimental Ideas

#### 10.1 Relational input-output pairs

Input-output pairs in ISP are abstracted as program states, which we can generate embeddings for. If the program attributes that need to be estimated were to be relational, then an RN could in theory improve the MLP used for estimating the program attributes. This item would be worth testing in a setup where relational program attributes could somehow be used to optimize the program search.

#### 10.2 Program attributes as questions

The cognitive process of designing a program to solve a problem is a highly complex task. Often times, it is a longer interactive process during which the solver has to ask a series of questions in order to arrive at the right programming technique and abstractions through meaningful decisions. This item aims to address the topic of how questions could be generated in a programming setup. The ability to ask meaningful questions seems to be a necessary component when trying to design a more general reasoning system.

#### 10.3 SortNet (rel) vs. RN (& debate on how MLP would implement logic in RN)

#### 10.4 A more general comparison of relational and composite machine learning models