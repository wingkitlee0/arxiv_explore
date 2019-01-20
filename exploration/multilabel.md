# Multi-class classification for text (single label)

## Introduction

This project is to classify the abstract/summary into some sub-category of astronomy

## Data

```texts = list of abstracts```

```labels = list of labels```

## Steps
1. Tokenize
    1. Counting the N most-frequent word, where N can be 10000.
    2. build a dictionary and represent each word with an integer.
    3. each abstract (a list of sentences) can be converted into a sequence of integer.
2. Embedding
    1. This is basically the first layer of the neural network
    2. It projects the tokenized sentences (list of integers) onto the embedding space, which usually has a smaller dimension than 
    2. 
3. 

## Data preparation

### Tokenization

**Tokenization** consists of two steps. The first step is to build a dictionary based on the text; the second step is to convert the text into a list of integers based on the dictionary.

#### Building dictiionary
To build the dictionary, we can use:
```
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
```
Here `max_words` is the maximum size of the dictionary **during conversion** (see the discussion [here](https://github.com/keras-team/keras/issues/7551
)). This means that the Tokenizer still stores every unique word it encounters during the `fit_on_texts()` routiner. The full dictionary can be found using `tokenizer.word_index`, which returns a python dictionary (e.g., `{'astronomy' : 10}`). The maximum possible integer can be larger than `max_words` (of course, the resultant dictionary can be smaller, depending on the type of document).

#### Conversion to integer sequence
To convert the list `texts` to a list of sequences, we can use:
```
sequences = tokenizer.texts_to_sequences(texts)
```
Here `sequences` is a list. Each element corresponds to a separate abstract, which is now a list of integers. Now, this *maximum allowed integer* is given by `max_words-1`. For example, a sentence "We found a new planet!" becomes:

```python
print(tokenizer.texts_to_sequences(["We found a new planet"]))
>>> [[7, 116, 4, 75, 272]]
```

Since each sequence has different length. We use the following to do the *padding* with zeros.

```
data = pad_sequences(sequences=sequences, maxlen=maxlen)
```
where `maxlen` is the maximum length of the sequence.

## Embedding

### Preparation

A common strategy is to embed the integer sequence into a smaller dimension *embedding*. This process converts each integer (or a word, before *tokenizing*) into a new vector. Two vectors with similar or related meanings will have a small *distance* (magnitude of the difference). For example, the words "planet" and "moon" should be closer than "inflation".

#### GloVe

We can easily see that such embedding --- representing relations between words --- requires a huge effort to construct. Therefore, we use a ready-to-use embedding called GloVe for this project.

Assuming we have already downloaded the GloVe vectors, we can read by the following code:

```python
import os
glove_dir = "../glove.6B/"

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print("Found %s word vectors." % len(embeddings_index))
```

Here the embedding dimension is 100 (other numbers are also available). In this case, the length of the `embeddings_index` is 400000. 

The `embeddings_index` is basically another dictionary that converts a word into embeddings vectors (of dimension 100). This time each vector (representing a single word) is an array of real numbers. To compare the embedding distance between two words, we define the following functions:

```python
def compare_words(w1, w2):
    v1 = embeddings_index.get(w1)
    v2 = embeddings_index.get(w2)
    if v1 is not None and v2 is not None:
        distance = np.sqrt(np.sum(v1-v2))
        print("embedding distance between %s and %s is %f" % (w1, w2, distance))
    else:
        if v1 is None:
            print("%s does not exist in the embeddings." % w1)
        if v2 is None:
            print("%s does not exist in the embeddings." % w2)
```

We can now compare the distances among "planet", "moon", and "inflation":

```python
compare_words("planet", "moon")
compare_words("planet", "inflation")
compare_words("moon", "inflation")
```
The output is
```
embedding distance between planet and moon is 2.600663
embedding distance between planet and inflation is 4.695775
embedding distance between moon and inflation is 3.909841
```

Next, we need to build the (dense) *matrix* that converts our tokenized integer sequences into the embedding:

```python
embeddings_dim = 100 # embedding dimension. It should correspond to the dimension used in the GloVe embedding above.

embedding_matrix = np.zeros((max_words, embeddings_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
```

Now, the `embedding_matrix` is a 10000 x 100 matrix. The first dimension is `max_words`. These are the top 10000 words that are in all the data. In our tokenized data, such word-index has to be $\leq$ `max_words` and so each word can be mapped into a 100-vector.

## Neural Network

After preparing the data and embedding, we can start to construct our neural network. To do this, we need to understand the structure of our data.

### Data arrays

Recall that the tokenization converts each word in an abstract into an integer $n$ (that is $n \leq$ `max_words`). Also, after the padding procedure, each data set is of length $N_{max}$ (`maxlen`). The input is a basically an *integer* matrix of dimensions $N_{data} \times N_{max}$. 

### First two layers

The first layer is an **Embedding()* layer.

The embedding matrix transforms each integer into a vector of length `embeddings_dim`. Therefore, the *output* of the embedding layer has a dimension of $N_{data} \times N_{max} \times N_{embeddings}$. 

In order to feed this output to other layer, we need to *flatten* the array into lower dimensions (Note: I am not sure if it is possible treat the data as two-dimensional, like an image.). Thus, the next layer is a *Flatten()* layer. The output of this layer has a dimension of $N_{data} \times (N_{max} \times N_{embeddings})$. The second dimension is $N_{max} \times N_{embeddings}$, which is very very long. 


