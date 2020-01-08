#!/usr/bin/env python3

import pandas as pd
import re
import sys
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding
 
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
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGTH))
model.add(Conv1D(64, 15, activation='relu'))
model.add(MaxPooling1D(15))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=3, batch_size=512, verbose=1,
          validation_data=(X_train_sequences[-100:], y_train[-100:]))

X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)

scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.8766
  
fdT.close()
fdS.close()
