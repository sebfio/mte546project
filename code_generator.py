#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

import pandas as pd
import numpy as np
import os
import re
import signal
import sys

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
            dfText          = [i.lower() for i in dfTextLong]
            dfText          = [re.sub(r'[^a-z ]', '', i) for i in dfText]
            dfText          = [' '.join(item.split()[:max_words_review - 1]) for item in dfTextLong if item]
            dfSentiment     = fdS.readlines() 
            
            frequencyArray = np.zeros((len(dfText), max_words_review))
            for i, review in enumerate(dfText):
                for j, word in enumerate(review.split(' ')):
                    frequencyArray[i][j] = word_bank[word] if word in word_bank else 0

            y_train = np.array([1 if int(i) > 0 else 0 for i in dfSentiment])

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

model.add(Conv1D(60, 25, activation='relu'))
model.add(MaxPooling1D(25))
model.add(Flatten())
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# model.add(LSTM(50))
# model.add(Dropout(0.1))
# #model.add(Dense(60, activation='sigmoid'))
# model.add(Dense(500, activation='sigmoid'))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print(model.summary())

filesTrain  = [name for name in os.listdir(trainTextPath)]
stepsTrain  = len(filesTrain)
filesTest   = [name for name in os.listdir(testTextPath)]
stepsTest   = len(filesTest)

def handler(signum, frame):
    print("Got kill signal, saving model anyways")
    model.save('trainedFunnyYelp.neural_net')
    sys.exit(1)

signal.signal(signal.SIGINT, handler)

model.fit_generator(trainGen, 
                steps_per_epoch=stepsTrain,
                validation_data=testGen,
                validation_steps=stepsTest,
                epochs=2,
                shuffle=True,
                verbose=1)

model.save('trainedFunnyYelp.neural_net')

## show a nicely formatted classification report
#print("[INFO] evaluating network...")
#print(classification_report(testLabels.argmax(axis=1), predIdxs, target_names=lb.classes_))
#print("Accuracy:", scores[1]) # 0.8766
