## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 8. A Comparison of relational and compositional machine learning models (part 1)

Based on the literature studied so far (and the experiments performed in 2019) we can start highlighting some similarities between the deep learning models which aim to perform relational reasoning / inference. Generally, the shortcomings in capturing relational properties from a dataset by a machine learning model are due to the lack of proper design and/or prior knowledge. However, this problem manifests itself differently depending on the case. Let us take a look at 2 well known examples.

#### 8.1 Convolution Maps

In visual recognition tasks, CNNs outperform MLPs simply because they exploit spatial relationships. Instead of having fully connected layers with different learnable weights, CNNs have shared weights (kernels / convolutions) and thus learn locally invariant features, meaning that the same properties (edges, textures, etc.) are learnt across every region of the image, whereas a MLP would not have this constraint and thus would have the potential to overfit specific properties of a given region. For this reason, we can conclude that CNNs have the proper design to learn (fit) relations between pixels and regions in images which are generally (and not only locally) useful for computing the required output in a visual task.

#### 8.2 Attention Functions

The next example is a machine translation model, namely the sequence to sequence modelling, where a recurrent neural network is fed an input sequence and has to produce an output sequence, such as translating a sentence from english to french (Sutskever et al. [14]) or synthesizing a program from a problem description (Li et al. [15]). One major breakthrough in this area was the use of an attention function (Bahdanau et al. [16]). Various implementations of attention models (Shazeer [17]) and visual attention (Xu et al. [18]). The seq2seq model was initially designed as an encoder-decoder architecture, where an RNN would process the input and provide a vector / state for the decoder to decode into the output.

One problem with this model was that it was not capable to properly encode longer sentences into a finite hidden state (fixed length vector) at the end of processing the input. And so essential information would be lost this way. The attention function alters this behaviour by constraining the decoder to attend to the hidden states of the encoder in a finite subsequence around the target word (interval), thus providing a context at each step in the output sequence by utilizing potential relations between consecutive words. The practical consequence of this modification is an enhanced ability to correctly learn to generate larger sequences (improved generalization capabilities).

|Seq2seq|Info|
|:-:|:---------:|
|![Encoder-Decoder versions](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/encoder_decoder.png)|**Encoder-Decoder versions**. (a) Vanilla Encoder-Decoder: only the final hidden state of the encoder is passed on to the decoder as initial input. (b) Attention based Encoder-Decoder: intermediary hidden states from the encoder are weighted in according to an attention function and fed into the decoder at all steps.|
|![Attention function](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attention_function.png)|**Attention function**: a more detailed view of the mechanism and computation.|

|Attention matrix|Info|
|:-:|:---------:|
|![Attention matrix](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attention_matrix.png)|Source: Bahdanau et al. [16]. Displays how much should the hidden state obtained when processing the j-th english word contribute to predicting the i-th french word.|

|MAC Attention map|Info|
|:-:|:---------:|
|![MAC Attention map](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/mac_attention.png)|Source: Hudson & Manning [4]. 3-steps reasoning based on the interaction of the memory, attention and control units.|

#### 8.3 Tutorials and supporting documentation

[A. LSTM and Attention](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)

[B. Encoder-Decoder Attention](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)

[C. Andrew Ng Tutorials](https://www.youtube.com/watch?v=RLWuzLLSIgw)

[D. Recurrent Relational Networks](https://paperswithcode.com/paper/recurrent-relational-networks)

### 9. A Comparison of relational and compositional machine learning models (part 2)

#### 9.1 SortNet vs. RN (Santoro et al. [5])

**SortNet steps** (2019 experiments)

- Design: eg. using an adequate neural network structure
- Prior Knowledge: eg. using the order relations as input
- Relational Network: Design + Prior Knowledge - integrating the prior knowledge into a neural network's design without data transformation 

**Problem Formulation**

**Comparison**

Both the SortNet and the RN learn relations between objects in the input, but the learning is modelled in a slightly different way.

- Using convolutions to represent relations in the SortNet case.
- Using pairs as separate training data for a relational function in the RN case.

Debate on how MLP would implement logic in RN.

#### 9.2 RN (Santoro et al. [5]) vs MAC (Hudson & Manning [4])

The relational function learned at the level of paired objects (CNN feature maps) in the RN model is very similar at the conceptual level with the attention function learned in the seq2seq model for pairs of words in different languages.

### 10. Relational reasoning and question answering in programming

#### 10.1 Relational input-output pairs

Consider input-output pairs in IPS to be program states, which we can generate embeddings for. If the program attributes that need to be estimated were to be relational, then an RN could in theory improve the MLP used for estimating the program attributes. This item would be worth testing in a setup where relational program attributes could somehow be used to optimize the program search.

#### 10.2 Program attributes as questions

The cognitive process of designing a program to solve a problem is a highly complex task. Often times, it is a longer interactive process during which the solver has to ask a series of questions in order to arrive at the right programming technique and abstractions through meaningful decisions. Thus, the ability to ask meaningful questions seems to be a necessary component when trying to design a more general reasoning system.

How are questions generated in a programming setup?

### 11. Directions in Program Induction

