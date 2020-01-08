#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

import pandas as pd
import re
import sys
import numpy as np
from numpy import array
import os
import textwrap

vocab_size = 10000
max_words_review = 100

def create_embedding_matrix(filepath, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    wordBank = dict()
    wordsProcessed = 1
    with open(filepath) as f:
        for line in f:
            if wordsProcessed > vocab_size - 1:
                break

            # Split the word with the numbers
            word, *vector = line.split()
            # Add token ID
            wordBank[word]=wordsProcessed

            embedding_matrix[wordsProcessed] = np.array(
                vector, dtype=np.float32)[:embedding_dim]

            wordsProcessed += 1

    return embedding_matrix, wordBank

embedding_dim = 50
embedding_matrix, word_bank = create_embedding_matrix('glove/glove.6B.50d.txt', embedding_dim)

json_file = open(sys.argv[1], 'r')
loaded_json = json_file.read()
json_file.close()
model = model_from_json(loaded_json)
print("Loading model %s" % (sys.argv[1]))

model.load_weights(sys.argv[2])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Loading model with weights %s" % (sys.argv[2]))
print("Evaluating sentence %s" % sys.argv[3])

frequencyArray = np.zeros((1,max_words_review))
for j, word in enumerate(sys.argv[3].split(' ')):
    if j >= max_words_review:
        break
    frequencyArray[0][j] = word_bank[word] if word in word_bank else 0

predict = model.predict(frequencyArray)
print(predict[0][0])
