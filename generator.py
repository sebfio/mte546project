import numpy as np
import keras
import os

#Get Item called by generator when batch corresponding to a given index is called
#Get item calls data generation with the list of IDs
#Data Generation action makes the data into numpy arrays
class DataGenerator(keras.utils.Sequence):
	def __init__(self, list_IDs, labels, batch_size, dim, n_channels=1,
             n_classes=10, shuffle=True):
	    'Initialization'
	    self.dim = dim
	    self.batch_size = batch_size
	    self.labels = labels
	    self.list_IDs = list_IDs
	    self.n_channels = n_channels
	    self.n_classes = n_classes
	    self.shuffle = shuffle
	    self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def on_epoch_end(self):
	 	'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
	    	np.random.shuffle(self.indexes)

		
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] #Basically grabs the indexes from batch location to the next batch location	

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes] #so for each index, get the lists_IDs at that index and save them to list_IDs_temp?

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
	  	# Initialization
	  	X = np.empty((self.batch_size, *self.dim, self.n_channels)) #These are inputs/outputs? Like X would be text and y would be funny/not?
	  	y = np.empty((self.batch_size), dtype=int)

	  	# Generate data
	  	for i, ID in enumerate(list_IDs_temp):
	      fdT = open('data/split_data/reviewText_' + str(i))
	      fdS = open('data/split_data/reviewFunny_' + str(i))
	  	  dfTextBin      	= fdT.readlines() 
		  dfText      	= [str(i) for i in dfTextBin]
		  dfSentiment 	= fdS.readlines() 

		  X_train = np.array(aX_train)
		  X_test  = np.array(aX_test)
		  y_train = np.array(ay_train)
		  y_test  = np.array(ay_test)
	  	  # Store sample
	  	  #X[i,] = np.loadtxt('data/split_data/reviewText_' + i) #Need this to match our file system

	      # Store class
	      y[i] = self.labels[ID] #Get label from list of labels which is found in the funny files

	  	return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


	