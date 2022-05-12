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
|![MAC Attention map](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/mac_attention.png)|Source: Hudson & Manning [4]. 3-steps reasoning (MAC network of length 3) based on the interaction of the memory, attention and control units.|

#### 8.3 Tutorials and supporting documentation

[A. LSTM, Encoder-Decoder and Attention](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)[+](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)

[B. Andrew Ng Tutorials](https://www.youtube.com/watch?v=RLWuzLLSIgw)

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

The cognitive process of designing a program to solve a problem is a highly complex task. Often times, it is a longer interactive process during which the solver has to ask a series of questions in order to arrive at the right programming technique and abstractions through meaningful decisions. Thus, the ability to ask meaningful questions seems to be a necessary component when trying to design a more general reasoning system. To research how questions could be generated in a programming setup?

### 11. Directions in Program Induction

#### 11.1 Latent Induction vs Code Synthesis

The first important bifurcation in the approaches for generating programs is the choice of program representation. When a neural network learns to map input to output, thus solving a programming task, the program is stored in the network and executed by the network through neural activations. This is called latent program induction, because the representation of the generated program is not human-readable.

The second choice is to formulate the problem such that the neural network outputs a program in a language, which is then executed to get the desired output. This is generally referred to as program synthesis.

A comparison of the two approaches applied on string transformation problems is carried in Devlin et al. [21].

Latent programs are written in the language of neural networks, whereas synthesized programs are written in a language of choice. Both approaches have shown success, however it is not possible to pick one that works best because they have different strengths. For instance, induction is more likely to provide a good approximation of the output function for the type of inputs provided, but might not generalize so well for new inputs. On the other hand, synthesis will either find the correct program and generalize the solution well for all inputs, or find the wrong solution which over-fits the presented input. Synthesis is thus more capable, but also the riskier approach.

#### 11.2 Specifications vs Input-Output Pairs

The second important ramification in formulating a program learning task is based on how the problem is conveyed to the network. Two directions are currently being extensively researched, one is to have specifications for solving the problem in natural language, the other is based on feeding the model with many input-output pairs.

There are also hybrid methods, where both types of information are presented to the learning model. While Yin and Neubig [10] present a method for inferring code from specifications, Balog et al. [7] and Parisotto et al., [23] perform program synthesis based on input-output pairs. The methods in (Ling et al., 2016 add ref) and (Ling et al., 2017 add ref) are examples of hybrid approaches.

#### 11.3 End-to-End Learning vs Intermediate Steps Prediction