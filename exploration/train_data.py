import numpy as np 
import tensorflow as tf
import pandas
import sys

target_name_dict = {'astro-ph.GA' : 0,
                    'astro-ph.SR' : 1,
                    'astro-ph.IM' : 2,
                    'astro-ph.EP' : 3,
                    'astro-ph.HE' : 4,
                    'astro-ph.CO' : 5
                }
label2target = { v:k for k,v in target_name_dict.items()}

files = ["data/2014astroph_p.h5",
         "data/2015astroph_p.h5",
        ]

abstracts = []
labels = []
for f in files:

    store = pandas.HDFStore(f)
    df = store['/df']
    store.close()

    abstracts += list(df['abstract'])
    labels = np.hstack([labels,np.array(df['label'])])


for i in range(2):
    print(abstracts[i])
    print( label2target[labels[i]] )
    print("---------")

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(abstracts)
sequences = tokenizer.texts_to_sequences(abstracts)
seq = pad_sequences(sequences, padding='post', value=0, maxlen=100)

np.random.seed(1234)
ind = np.random.randint(0, len(labels), len(labels))
print(ind.shape)
labels = labels[ind]
seq = seq[ind,:]


split_1 = int(0.8 * len(labels))
split_2 = int(0.9 * len(labels))
train_labels = labels[:split_1]
dev_labels = labels[split_1:split_2]
test_labels = labels[split_2:]

train_seq = seq[:split_1, :]
dev_seq = seq[split_1:split_2, :]
test_seq = seq[split_2:, :]


#%%
vocab_size = 10000


#%%
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(6, activation=tf.nn.sigmoid))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.utils import to_categorical
train_labels_onehot = to_categorical(train_labels)
test_labels_onehot = to_categorical(test_labels)

history = model.fit(train_seq, train_labels_onehot, epochs=10, steps_per_epoch=32, validation_split=0.3, validation_steps=32)


#%%
model.evaluate(test_seq, test_labels_onehot, steps=10)


#%%
text = ["we present high dispersion spectroscopic data of the compact planetary nebula vy 1 2 where high expansion velocities up to 100 km s are found in the ha n ii and o iii emission lines hst images reveal a bipolar structure vy 1 2 displays a bright ring like structure with a size of 2 4 2 and two faint bipolar lobes in the west east direction a faint pair of knots is also found located almost on opposite sides of the nebula at pa degrees furthermore deep low dispersion spectra are also presented and several emission lines are detected for the first time in this nebula such as the doublet cl iii a k iv a c ii 6461 a the doublet c iv 5801 5812 a by comparison with the solar abundances we find enhanced n depleted c and solar o the central star must have experienced the hot bottom burning cn cycle during the 2nd dredge up phase implying a progenitor star of higher than 3 solar masses the ver"]


#%%
seq_1 = tokenizer.texts_to_sequences(text)


#%%
seq_2 = pad_sequences(seq_1, padding='post', value=0, maxlen=350)


#%%
prob = model.predict(seq_2)
prob /= prob.sum()
print(prob)
ii = np.argmax(prob)
print(label2target[ii])





