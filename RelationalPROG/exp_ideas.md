## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 9. Experiments and evaluation of results (part 1)

#### 9.1 A Comparison of relational and compositional machine learning models

Based on the literature studied so far (and the experiments performed in 2019) we can start highlighting some similarities between the deep learning models which aim to perform relational reasoning / inference. Generally, the shortcomings in capturing relational properties from a dataset by a machine learning model are due to the lack of proper design and/or prior knowledge. However, this problem manifests itself differently depending on the case. Let us take a look at 2 well known examples.

##### 9.1.1 Convolutions

In visual recognition tasks, CNNs outperform MLPs simply because they exploit spatial relationships. Instead of having fully connected layers with different learnable weights, CNNs have shared weights (kernels / convolutions) and thus learn locally invariant features, meaning that the same properties (edges, textures, etc.) are learnt across every region of the image, whereas a MLP would not have this constraint and thus would have the potential to overfit specific properties of a given region. So we could state that CNNs have the proper design to learn (fit) relations between pixels and regions in images which are generally (and not only locally) useful for computing the required output in a visual task.

##### 9.1.2 Attention

The next example is a machine translation model, namely the sequence to sequence modelling, where a recurrent neural network is fed an input sequence and has to produce an output sequence, such as translating a sentence from english to french (see 10.1) or synthesizing a program from a problem description (see 10.2). One major breakthrough in this area was the use of an attention function (see 10.3). Various implementations of attention models (see 10.4) and visual attention (see 10.5). The seq2seq model was initially designed as an encoder-decoder architecture, where an RNN would process the input and provide a vector / state for the decoder to decode into the output.

One problem with this model was that it was not capable to properly encode longer sentences into a finite hidden state at the end of processing the input. And so essential information would be lost this way. The attention function alters this behaviour by constraining the decoder to attend to the hidden states of the encoder in a finite subsequence around the target word (interval), thus providing a context at each step in the output sequence by utilizing potential relations between consecutive words. The practical consequence of this modification is an enhanced ability to correctly learn to generate larger sequences (improved generalization capabilities).

|Img|Description|
|:-:|:---------:|
|![Encoder-Decoder versions](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/encoder_decoder.png)|Encoder-Decoder versions|
|![Attention function](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attention_function.png)|Attention function|

|Img|Description|
|:-:|:---------:|
|![Attention matrix](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/attention_matrix.png)|Attention matrix|
|![MAC Attention map](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/mac_attention.png)|MAC Attention map|

##### 9.1.3 SortNet (rel) vs. RN

SortNet steps (2019 experiments)

- Design: eg. using an adequate neural network structure
- Prior Knowledge: eg. using the order relations as input
- Relational Network: Design + Prior Knowledge

Debate on how MLP would implement logic in RN.

##### 9.1.4 RN vs MAC

#### 9.2 Relational reasoning and question answering in programming

##### 9.2.1 Relational input-output pairs

Consider input-output pairs in IPS to be program states, which we can generate embeddings for. If the program attributes that need to be estimated were to be relational, then an RN could in theory improve the MLP used for estimating the program attributes. This item would be worth testing in a setup where relational program attributes could somehow be used to optimize the program search.

##### 9.2.2 Program attributes as questions

The cognitive process of designing a program to solve a problem is a highly complex task. Often times, it is a longer interactive process during which the solver has to ask a series of questions in order to arrive at the right programming technique and abstractions through meaningful decisions. Thus, the ability to ask meaningful questions seems to be a necessary component when trying to design a more general reasoning system.

How are questions generated in a programming setup?

### 10. Additional studies / learning material

#### More imprtant papers

[1. Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

[2. Competition-Level Code Generation with AlphaCode](https://storage.googleapis.com/deepmind-media/AlphaCode/competition_level_code_generation_with_alphacode.pdf) [+](https://www.deepmind.com/blog/competitive-programming-with-alphacode)

[3. Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

[4. Fast Transformer Decoding](https://arxiv.org/pdf/1911.02150.pdf)

[5. Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

#### Tutorials and supporting documentation

[A. LSTM and Attention](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)

[B. Encoder-Decoder Attention](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)

[C. Recurrent Relational Networks](https://paperswithcode.com/paper/recurrent-relational-networks)

[D. Andrew Ng Tutorials](https://www.youtube.com/watch?v=RLWuzLLSIgw)