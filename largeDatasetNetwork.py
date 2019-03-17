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


def revectorize(model):
	#Need to add data
	model.add(Embedding(len(training_generator.vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGTH))


def printResults():
	print("result")
params = {'dim': (10000,1),
          'batch_size': 5,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': False}



#Need partition dictionary which contains the training and the validation ids
#reviewFiles = open(sys.argv[1], mode='rb')
#unnyFiles = open(sys.argv[2], mode='r')

#aX_train, aX_test, ay_train, ay_test = train_test_split(reviewFiles, funnyFiles, test_size=0.2)

#callback = [revectorize(model), printResults()]
testIds = []
trainIds  = list(range(0,500))
trainIds = []
testIds = list(range(501,669))
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)


training_generator = DataGenerator(list_IDs=trainIds,vectorizer=vectorizer,train=1, **params)
validation_generator = DataGenerator(list_IDs=testIds,vectorizer=vectorizer,train=0, **params)

X_train_onehot = training_generator.vectorizer.fit_transform(training_generator[0])

word2idx = {word: idx for idx, word in enumerate(training_generator.vectorizer.get_feature_names())}
tokenize = training_generator.vectorizer.build_tokenizer()
preprocess = training_generator.vectorizer.build_preprocessor()
 
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes
 
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_train]


model = Sequential()
model.add(Embedding(len(training_generator.vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGTH))
model.add(Conv1D(64, 15, activation='relu'))
model.add(MaxPooling1D(15))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #They use a different one in the tutorial, rms or something


model.fit_generator(generator=training_generator,callbacks=[revectorize(model)], validation_data=validation_generator, use_multiprocessing=True, workers=4)