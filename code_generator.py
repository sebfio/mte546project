#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import pandas as pd
import re
import sys
import numpy as np
from numpy import array
import os
import textwrap

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
#
#from nltk.corpus import stopwords
#
#from keras.models import Sequential
#from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding
#from keras.preprocessing.text import one_hot
#from keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing.text import Tokenizer

MAX_LENGTH_SEQ  = 180 
BS              = 50
NUM_PER_FILE    = 5000

vocab_size=10000

max_words_review = 300

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
            dfText          = [' '.join(item.split()[:max_words_review]) for item in dfTextLong if item]
            dfSentiment     = fdS.readlines() 

            #print("============")
            #print(len(dfText))
            #print(len(dfSentiment))
            
            X_train = np.array(dfText)
            y_train = np.array(dfSentiment)

            padded_X_train = sequence.pad_sequences(X_train, maxlen=max_words_review)

            fdT.close()
            fdS.close()

            yield(padded_X_train, y_train)
        
## Get functors for generation of train and test data
trainTextPath   = "data/trainText"
trainFunnyPath  = "data/trainFunny"
testTextPath    = "data/testText"
testFunnyPath   = "data/testFunny"
trainGen    = dataGenerator(trainTextPath, trainFunnyPath)
testGen     = dataGenerator(testTextPath, testFunnyPath)

model = Sequential()
embedding_vector_length = 50 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_words_review))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

filesTrain  = [name for name in os.listdir(trainTextPath)]
stepsTrain  = len(filesTrain)
filesTest   = [name for name in os.listdir(testTextPath)]
stepsTest   = len(filesTest)
     
model.fit_generator(trainGen, 
                steps_per_epoch=stepsTrain,
                validation_data=testGen,
                validation_steps=stepsTest,
                epochs=3)

model.save('trainedFunnyYelp.neural_net')

## show a nicely formatted classification report
#print("[INFO] evaluating network...")
#print(classification_report(testLabels.argmax(axis=1), predIdxs, target_names=lb.classes_))
#print("Accuracy:", scores[1]) # 0.8766
