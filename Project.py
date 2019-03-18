#!/usr/bin/python3
# Two Methods for natural language processing that I can see

# 1. BOW (BAg of Words) which basically treats the text as an unrelated bag of words. Key is that it knows nothing about semantics. Downside is vectors here are sparse. Words just don't repeat all that often for the most part, and 
# we use different slots for related/similar words
# 2. A better method is using vectors (AKA word vectors). Here words are classified somewhat semantically. Two conditions should be met for this representation
#  - Similar words should be close together
#  - Allow word analogies "King" - "man" + "woman" = "queen"
# Related to RNNs
# Idea is that king and woman are related closely on one axis, but king and man are associated closely on another
# Word2Vec is one of the more popular algorithms for word vectorization
# Basically BOW looks for word 0 given words -3,-2,-1,1,2,3. 
# Word2Vec looks for words -3,-2,-1,1,2,3 given word 0.

# 3. Word2Vec Models
# Gensim is one that's pretty good
# # Go here and download + unzip the Text8 Corpus: http://mattmahoney.net/dc/text8.zip
# # We take only words that appear more than 150 times for doing a visualization later
# w2v_model2 = Word2Vec(Text8Corpus('~/Downloads/text8'), size=100, window=5, min_count=150, workers=4)

# Here for example it looks at the text8 dataset and extracts words that appear 150 times or more

# if we want to visualize it we use t-sne, which compresses it to 2d/3d (wahey look at me learning SYDE522 by accident)

# 4. fastText
# Facebook's implementation, also very good but much slower than the gaensim one because it looks at substrings of words
#!/usr/bin/env python3

import pandas as pd
import re
import sys
import numpy as np
from generator import DataGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding

import dask.dataframe as dd
import itertools

#N = [2,4]
# with open("data/reviewText.txt") as myfile:
#     head = itertools.islice(myfile,i,)
# print(head[0])

# #for i in range(1, 2):
# chunksize=100
# reviewChunk = pd.read_csv('data/reviewTextTiny.txt', engine='python', chunksize=chunksize, delimiter='\n')
# funnyChunk = pd.read_csv('data/reviewFunnyTiny.txt',  engine='python',chunksize=chunksize,delimiter='\n',)
# combinedDf = pd.DataFrame([reviewChunk,funnyChunk]) #Much faster than appending a dataframe every time

def generate_batches(reviewFiles, funnyFiles, batch_size):
   counter = 0
   while True:
     reviewName = reviewFiles[counter]
     funnyName = funnyFiles[counter]
     print(reviewName)
     print(funnyName)
     counter = (counter + 1) % len(files)
     data_bundleX = pickle.load(open(reviewName, "rb"))
     data_bundleY= pickle.load(open(funnyName, "rb"))
     X_train = data_bundleX[0]
     y_train = data_bundleY[1].astype(np.float32)
     y_train = y_train.flatten()
     for cbatch in range(0, X_train.shape[0], batch_size):
         yield (X_train[cbatch:(cbatch + batch_size),:,:], y_train[cbatch:(cbatch + batch_size)])

# train_files = [train_bundle_loc + "bundle_" + cb.__str__() for cb in range(nb_train_bundles)]
# gen = generate_batches(files=train_files, batch_size=batch_size)
# history = model.fit_generator(gen, samples_per_epoch=samples_per_epoch, nb_epoch=num_epoch,verbose=1, class_weight=class_weights)
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x));

# Process data, get a file descriptor for the input and output
fdT         = open(sys.argv[1], mode='rb')
fdS         = open(sys.argv[2], mode='r')

# Read data in and out into a array/matrix like form
dfTextBin      	= fdT.readlines() 
dfText      	= [str(i) for i in dfTextBin]
dfSentiment 	= fdS.readlines() 

# Shuffle the data and then split it, keeping 20% aside for testing
#aX_train, aX_test, ay_train, ay_test = train_test_split(dfText, dfSentiment, test_size=0.2)

 #Size of training sets
numBatches = len(dfTextBin)
len(dfTextBin)


aX_train, aX_test, ay_train, ay_test = train_test_split(dfText, dfSentiment, test_size=0.2)
X_train = np.array(aX_train)
X_test  = np.array(aX_test)
y_train = np.array(ay_train)
y_test  = np.array(ay_test)

vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)

X_train_onehot = vectorizer.fit_transform(X_train)

## NETWORK MAKING
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()
 
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes
 
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_train]


## Compute the max lenght of a text
MAX_SEQ_LENGTH = len(max(X_train_sequences, key=len))
#print("MAX_SEQ_LENGTH=", MAX_SEQ_LENGTH)
 
from keras.preprocessing.sequence import pad_sequences
N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
#print(X_train_sequences[0])

model = Sequential()
## Model architecture
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGTH))
model.add(Conv1D(64, 15, activation='relu'))
model.add(MaxPooling1D(15))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

## Generate model architecture
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batchSize = 512


model.fit(X_train_sequences[:-100], y_train[:-100], #Fit set for small datasets
          epochs=3, batch_size=batchSize, verbose=1,
          validation_data=(X_train_sequences[-100:], y_train[-100:]))


train_files = [train_bundle_loc + "bundle_" + cb.__str__() for cb in range(nb_train_bundles)]

X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)

scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.8766

fdT.close()
fdS.close()


