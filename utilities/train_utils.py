import numpy as np
import tensorflow as tf
import random
import shelve
import _pickle as pickle
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Add, Reshape
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import np_utils, plot_model
from keras.losses import mean_squared_error


"""	Utilities for Training:

	- Loads the data Given a full path to the .npy
	- Trains a Compiled Model (single input or two inputs) from Run Utilities
	
	Note: X input data not normalized (e.g. to preserve coherence/interpretability (cost in chf, time in min, etc..) 
		  Q input data (=extra) normalized before training
"""


def train_sameInput(fileInputName, nEpoch, compiledModel, train_data_name, batchSize = 50, filePath = '',
					saveExtension = '', filePart = '', callback = None, validationRatio = 0, Hrusch=False , verbose = 0):

	train_data = np.load(train_data_name)
	train_labels = train_data[:,-1,:]
	train_data = np.delete(train_data, -1, axis = 1)

	train_data = np.expand_dims(train_data, -1)
#	train_data = normalize(train_data)

	if not Hrusch:
		history = fitModel(train_data, train_labels, nEpoch,batchSize,compiledModel,callback,validationRatio,verbose=verbose)
	else:
		#Each Utility has its own input layer to share the first dense network
		history = fitModel([np.expand_dims(train_data[:,:,i],-1) for i in range(train_data.shape[2])], train_labels,
						   nEpoch,batchSize,compiledModel,callback,validationRatio,verbose=verbose)

	#with open(filePath + 'trainHistoryDict' + saveExtension, 'wb') as file_pi:
	#	pickle.dump(history.history, file_pi)

	compiledModel.save(filePath + fileInputName +'_' + saveExtension + '.h5')



def train_extraInput(fileInputName, nEpoch, compiledModel, train_data_name, batchSize = 50, filePath = '',
					 saveExtension = '', filePart = '', callback = None, validationRatio = 0, NN = False , verbose = 0):

	train_data = np.load(train_data_name)  
	train_labels = train_data[:,-1,:]
	train_data = np.delete(train_data, -1, axis = 1)

	train_data = np.expand_dims(train_data, -1)


	extra_data = np.load(train_data_name[:-4] + '_extra.npy')
	nExtraFeatures = extra_data[0].size
	extra_data = np.expand_dims(np.expand_dims(extra_data, -1),-1)

	#train_data = normalize(train_data)
	extra_data = normalize(extra_data)

	if not NN:
		history = fitModel([train_data, extra_data], train_labels, nEpoch, batchSize,compiledModel,callback,validationRatio, verbose = verbose)
	else:
		history = fitModel(extra_data, train_labels, nEpoch, batchSize,compiledModel,callback,validationRatio, verbose = verbose)

	#with open(filePath + 'trainHistoryDict' + saveExtension, 'wb') as file_pi:
	#	pickle.dump(history.history, file_pi)

	compiledModel.save(filePath +fileInputName + '_' + saveExtension + '.h5')


def fitModel(train_data, train_labels, nEpoch, batchSize, model, callback, validationRatio, verbose = 0):
	""" Call fit function. Different calls whether use of Callback or validationRatio"""
	if callback is None: 
		if validationRatio == 0:
			history = model.fit(train_data,train_labels, epochs = nEpoch, steps_per_epoch = batchSize, verbose = verbose)
		else:
			history = model.fit(train_data,train_labels, epochs = nEpoch, validation_split = validationRatio,
								steps_per_epoch = batchSize, verbose = verbose, validation_steps = batchSize)
	else:
		if validationRatio == 0:
			history = model.fit(train_data,train_labels, epochs = nEpoch, steps_per_epoch = batchSize, verbose = verbose, callbacks = [callback])
		else:
			history = model.fit(train_data,train_labels, epochs = nEpoch, validation_split = validationRatio, steps_per_epoch = batchSize,
								verbose = verbose, validation_steps = batchSize, callbacks = [callback])
	return history



def saveBetas(fileInputName, model, filePath = '', saveExtension = '', verbose = True):
	betas_layer = model.get_layer(name = 'Utilities')
	betas = betas_layer.get_weights()
	if verbose:
		print(betas)
	np.save(filePath + 'Betas_'+ fileInputName + '_' + saveExtension +'.npy', betas)
	return betas

def normalize(data):
	return (data-data.mean(axis=0))/(data.std(axis=0))
