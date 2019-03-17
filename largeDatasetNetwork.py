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


params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}



#Need partition dictionary which contains the training and the validation ids
#reviewFiles = open(sys.argv[1], mode='rb')
#unnyFiles = open(sys.argv[2], mode='r')

#aX_train, aX_test, ay_train, ay_test = train_test_split(reviewFiles, funnyFiles, test_size=0.2)

vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)

model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGTH))
model.add(Conv1D(64, 15, activation='relu'))
model.add(MaxPooling1D(15))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #They use a different one in the tutorial, rms or something

training_generator = DataGenerator(partition['train'], labels, **params,train=1)
validation_generator = DataGenerator(partition['validation'], labels, **params, train=0)



model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=4)