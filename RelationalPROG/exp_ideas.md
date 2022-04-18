## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 9. Additional Studies

#### 9.1 More imprtant papers

[1. Competition-Level Code Generation with AlphaCode](https://storage.googleapis.com/deepmind-media/AlphaCode/competition_level_code_generation_with_alphacode.pdf)

[2. Fast Transformer Decoding](https://arxiv.org/pdf/1911.02150.pdf)

[3. Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

[4. Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

[5. Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

#### 9.2 Tutorials and supporting documentation

[1. LSTM and Attention](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)

[2. Encoder Decoder Attention](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)

[3. Recurrent Relational Networks](https://paperswithcode.com/paper/recurrent-relational-networks)

[4. Andrew Ng](https://www.youtube.com/watch?v=RLWuzLLSIgw)

### 10. Experimental Ideas

#### 10.1 Relational input-output pairs

Consider input-output pairs in IPS to be program states, which we can generate embeddings for. If the program attributes that need to be estimated were to be relational, then an RN could in theory improve the MLP used for estimating the program attributes. This item would be worth testing in a setup where relational program attributes could somehow be used to optimize the program search.

#### 10.2 Program attributes as questions

The cognitive process of designing a program to solve a problem is a highly complex task. Often times, it is a longer interactive process during which the solver has to ask a series of questions in order to arrive at the right programming technique and abstractions through meaningful decisions. Thus, the ability to ask meaningful questions seems to be a necessary component when trying to design a more general reasoning system.

How are questions generated in a programming setup? 

#### 10.3 SortNet (rel) vs. RN

Debate on how MLP would implement logic in RN.

#### 10.4 A more general comparison of relational and composite machine learning models

RN vs MAC