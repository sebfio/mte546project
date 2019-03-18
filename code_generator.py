#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import re
import sys
import numpy as np
from numpy import array
import os
import textwrap

MAX_LENGTH_SEQ  = 180 
BS              = 50
NUM_PER_FILE    = 5000

vocab_size = 7000

max_words_review = 300

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

def dataGenerator(filePathText, filePathFunny):
    filesText   = [name for name in os.listdir(filePathText)]
    filesFunny  = [name for name in os.listdir(filePathFunny)]

    assert(len(filesText) == len(filesFunny)) 
    while True:
        for i in range(len(filesText)):
            filename_structure = filesText[i].split("_")
            idFile = filename_structure[1] # Get the number after the underscore
            prefunny = filesFunny[0].split("_") # Get the funny file before the underscore
            filename_funny = prefunny[0] + "_" + idFile

            fullTextFileName = filePathText + "/" + filesText[i]
            fullFunnyFileName = filePathFunny + "/" + filename_funny

            print(fullTextFileName)
            print(fullFunnyFileName)

            fdT         = open(fullTextFileName, mode='rb')
            fdS         = open(fullFunnyFileName, mode='r')
            
            # Read data in and out into a array/matrix like form
            dfTextBin       = fdT.readlines() 
            dfTextLong      = [str(i) for i in dfTextBin]
            dfText          = [i.replace("\\", "") for i in dfTextLong]
            dfText          = [i.replace(".", " ") for i in dfTextLong]
            dfText          = [' '.join(item.split()[:max_words_review - 1]) for item in dfTextLong if item]
            dfSentiment     = fdS.readlines() 
            
            frequencyArray = np.zeros((len(dfText), max_words_review))
            for i, review in enumerate(dfText):
                for j, word in enumerate(review.split(' ')):
                    frequencyArray[i][j] = word_bank[word] if word in word_bank else 0

            y_train = np.array(dfSentiment)

            fdT.close()
            fdS.close()

            yield(frequencyArray, y_train)
        
## Get functors for generation of train and test data
trainTextPath   = "data/trainText"
trainFunnyPath  = "data/trainFunny"
testTextPath    = "data/testText"
testFunnyPath   = "data/testFunny"
trainGen    = dataGenerator(trainTextPath, trainFunnyPath)
testGen     = dataGenerator(testTextPath, testFunnyPath)

model = Sequential() 
model.add(Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_words_review))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_words_review))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

filesTrain  = [name for name in os.listdir(trainTextPath)]
stepsTrain  = len(filesTrain)
filesTest   = [name for name in os.listdir(testTextPath)]
stepsTest   = len(filesTest)
     
model.fit_generator(trainGen, 
                steps_per_epoch=stepsTrain,
                validation_data=testGen,
                validation_steps=stepsTest,
                epochs=3,
                shuffle=True,
                verbose=1)

model.save('trainedFunnyYelp.neural_net')

## show a nicely formatted classification report
#print("[INFO] evaluating network...")
#print(classification_report(testLabels.argmax(axis=1), predIdxs, target_names=lb.classes_))
#print("Accuracy:", scores[1]) # 0.8766