# Relational reasoning in deep learning: a parallel between solving visual and programming tasks

## 6. Design of models and connections

Based on the literature studied so far (and the experiments performed in 2019) we can start highlighting some similarities between the deep learning models which aim to perform relational reasoning / inference. Generally, the shortcomings in capturing relational properties from a dataset by a machine learning model are due to the lack of proper design and/or prior knowledge. However, this problem manifests itself differently depending on the case. Let us take a look at 2 well known examples.

### 6.1 Convolutional Maps (for Neural Networks)

In visual recognition tasks, CNNs (Convolutional Neural Networks) outperform MLPs (Dense Neural Networks) simply because they exploit spatial relationships. Instead of having fully connected layers with different learnable weights, CNNs have shared weights (kernels / convolutions) and thus learn locally invariant features, meaning that the same properties (edges, textures, etc.) are learnt across every region of the image, whereas a MLP would not have this constraint and thus would have the potential to overfit specific properties of a given region. 

Another advantage of learning fewer weights is the computational efficiency. For these reasons, we can gather that CNNs have the proper design to learn (fit) relations between pixels and regions in images which are generally (and not only locally) useful for inferring a solution in a visual task. Moreover, the feature maps computed in CNNs can be used as vector embeddings for images, which can be reused for various tasks.

**Strengths**:

- learning common weights
- generating image embeddings

[Conv Net 1](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

[Conv Net 2](https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

### 6.2 Attention Maps (for Sequence to Sequence Models)

The next example is a machine translation model, namely the sequence to sequence modelling, where a recurrent neural network is fed an input sequence and has to produce an output sequence, such as translating a sentence from english to french (Sutskever et al. [14]) or synthesizing a program from a problem description (Li et al. [15]). One major breakthrough in this area was the use of an attention function (Bahdanau et al. [16]). Since then, more variants and implementations have been developed, such as the attention models from (Shazeer [17]) and visual attention from (Xu et al. [18]). The seq2seq model was initially designed as an encoder-decoder architecture, where an RNN would process the input and provide a vector / state for the decoder to decode into the output.

One problem with this model was that it was not capable to properly encode longer sentences into a finite hidden state (fixed length vector) at the end of processing the input. And so essential information would be lost this way. The attention function alters this behaviour by constraining the decoder to attend to the hidden states of the encoder in a finite subsequence around the target word (interval), thus providing a context at each step in the output sequence by utilizing potential relations between consecutive words. The practical consequence of this modification is an enhanced ability to correctly learn to generate larger sequences (improved generalization capabilities).

**Strengths**:

- reducing information bottleneck in the encoder-decoder
- exploiting information relevance within a context 

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

#### Tutorials and supporting documentation

[A. LSTM, Encoder-Decoder and Attention](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)[+](https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb)

[B. More on Attention](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)[+](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)[+](https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/)

[C. Andrew Ng Tutorials](https://www.youtube.com/watch?v=RLWuzLLSIgw)

### 6.3 Relational Neural Networks

### 6.4 Graph Neural Networks

**What the graph can represent**:

- program
- molecule
- social network
- papers with citations

**Nodes**: information encoded into an embedding (vector state). Eg. image, word vectors, etc.

**Edges**: relations between nodes. Can be of multiple types.

**Output**: for each node it computes a state representing how the sample belongs to the overall graph.

|Example|Description|
|:-----:|:---------:|
|![Program](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/prog_graph.png)|TODO|
|![Convolution](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/conv_graph.png)|TODO|
|![Deep Sets](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/deep_sets.png)|TODO|

**Pseudocode**:

- look at current neighbouring states
- prepare messages
- summarize messages
- compute the next state of the node based on the current state and the neighbourhood summary

#### Tutorials and supporting documentation

[A. GNN implementation & intro](https://keras.io/examples/graph/gnn_citations/)[+](https://www.youtube.com/watch?v=2KRAOZIULzw)[+](https://www.youtube.com/watch?v=wJQQFUcHO5U)

[B. Lecture on GNNs](https://www.youtube.com/watch?v=zCEYiCxrL_0)